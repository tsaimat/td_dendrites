"""
Candidate bottom-up plasticity rules for the basal weights w_bas (shape: n_stimuli x n_neurons).

Every rule has the same signature and returns the *unbounded* weight update dw (same shape as w_bas):

    rule(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp, state) -> dw

    w_bas   current basal weights (n_stim, n_z)
    x_in    binary vector of currently active stimuli (n_stim,); all zeros at the outcome time step
    x_som   somatic activity of the pyramidal neurons (n_z,), = gains * x_bas
    x_bas   basal activation before apical gain, including background (n_z,)
    x_ap    apical activation in [1/MAX_GAIN, 1) (n_z,), the output of helper.apical_transfer
    gains   multiplicative apical gains in [1, MAX_GAIN] (n_z,), = helper.gain(x_ap)
    thetas  per-neuron plasticity thresholds (n_z,), theta_0 * expected basal drive per time step + BKG
    hp      hyperparameter dict; every rule reads hp['lr_bas'], rule-specific keys are documented per rule
    state   dict that persists across time steps and trials of one seed; slow variables (running averages,
            sliding thresholds) live here. Rules initialize their entries lazily on the first call.

The update is then applied by apply_bound() according to hp['bound'], so the post-synaptic factor (the rule) and
the mechanism that keeps the weights bounded are chosen independently:

    'soft'   w <- clamp(w + dw * (1 - w) * w, 0, 1)      the soft bound of the main code, weights stay in [0, 1]
    'clamp'  w <- clamp(w + dw, 0, 1)                    hard clamp only; total drive per neuron is unbounded
    'l1'     w <- clamp(rescale cols of clamp(w + dw, 0), 0, 1)  each neuron's afferents keep their initial total
                                                         (competition); then clamped to [0, 1]
    'none'   w <- w + dw                                 nothing (only sensible for rules with their own decay)
    'pre'    w <- clamp(rescale rows of clamp(w + dw, 0), 0, 1)  each stimulus keeps its initial total outgoing
                                                         weight (sum over neurons), so a rule can redistribute a
                                                         stimulus's drive across neurons but never grow it
    'both'   per-neuron rescaling (as 'l1') followed by 'pre', then clamped to [0, 1]
    Every bound keeps the weights in [0, 1] (the user's hard constraint for bottom-up plasticity).

The rules are called once per time step of the stimulus period and once at the outcome time step (with x_in = 0),
so slow variables also see the outcome-time apical activity.

Mapping of the literature rules (written in generic rate-model variables) onto this model
-------------------------------------------------------------------------------------------
The generic neuron is  x_som = gain(x_ap) * x_bas  with x_bas = w_bas . x_pre, which is exactly this model with
x_pre = x_in (binary stimuli), gain() = helper.gain, and the apical activation x_ap = helper.apical_transfer(...).
'dt' is one time step, so it is absorbed into lr_bas. The resting apical activation is apical_transfer(0) =
1/MAX_GAIN, where gain = 1.

Add a new rule by writing a function and registering it in RULES.
"""
import torch
from l5apical.helper import MAX_GAIN, BKG

A_REST = 1. / MAX_GAIN  # resting apical activation, apical_transfer(0); gain(A_REST) = 1


def apical_norm(x_ap: torch.Tensor) -> torch.Tensor:
    """Apical activation rescaled to [0, 1] (0 at rest, 1 at maximal gain), i.e. (gain - 1) / (MAX_GAIN - 1)."""
    return ((x_ap - A_REST) / (1 - A_REST)).clamp(0, 1)


def _pre_post(x_in: torch.Tensor, post: torch.Tensor, hp: dict) -> torch.Tensor:
    """Outer product of the presynaptic and post-synaptic factors, scaled by the basal learning rate."""
    return torch.outer(hp['lr_bas'] * x_in, post)


# -------------------------------------------------------------------------------------------------------------------
# Baselines
# -------------------------------------------------------------------------------------------------------------------

def none(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp, state):
    """No bottom-up plasticity (control)."""
    return None


def hebb(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp, state):
    """
    The rule currently in l5apical.simulations: Hebbian with the fixed thresholded post-synaptic term of the apical
    rule, x_in (x_som - theta). With bound = 'soft' this is bit-identical to Simulation.BU_PLASTICITY.
    Apical activity enters only through x_som = gain * x_bas.
    """
    return _pre_post(x_in, x_som - thetas, hp)


def gated_hebb(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp, state):
    """
    Three-factor variant of hebb: the thresholded Hebbian term is multiplied by the normalized apical activation,
    so basal synapses only change on neurons whose apical dendrite currently receives the top-down signal.
    hp['gate_power'] sharpens (> 1) or softens (< 1) the gate. hp['gh_post'] picks the post-synaptic variable that is
    compared with the threshold: 'som' (gain * x_bas, as in hebb) or 'bas' (the un-gained basal drive, so that the
    apical signal enters only through the gate and the threshold can separate a neuron's two texture weights even
    when its gain is high). hp['gh_theta_scale'] multiplies the threshold for this rule only (the apical rule keeps
    theta), so that the basal threshold can be tuned independently of the apical one.
    """
    gate = apical_norm(x_ap).pow(hp.get('gate_power', 1.0))
    post = x_bas if hp.get('gh_post', 'som') == 'bas' else x_som
    return _pre_post(x_in, gate * (post - hp.get('gh_theta_scale', 1.0) * thetas), hp)


def oja(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp, state):
    """
    Oja's rule: Hebbian growth minus a decay w * post^2 that bounds the L2 norm of each neuron's afferents.
    hp['oja_decay'] scales the decay (1.0 is the textbook rule). Meant to be used with bound = 'clamp' or 'none'.
    """
    return hp['lr_bas'] * (torch.outer(x_in, x_som) - hp.get('oja_decay', 1.0) * w_bas * x_som.pow(2))


# -------------------------------------------------------------------------------------------------------------------
# 1. Burst-dependent rule (Payeur et al., Nat Neurosci 2021)
# -------------------------------------------------------------------------------------------------------------------

def burst(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp, state):
    """
    dw = lr x_in x_som (P - Pbar).

    The burst fraction P is read off the apical activation: in the model the apical activation is the
    (sigmoidal) probability that the top-down input drives dendritic Ca2+ spikes and thereby switches the neuron to
    bursting, which is what raises the somatic gain. P uses the apical activation rescaled to 0 at rest
    (apical_norm, i.e. (gain - 1) / (MAX_GAIN - 1)), so P = 0 without apical input and also when the apical
    dendrite is inhibited (x_ap = 0 at the inhibited outcome step, below the resting apical_transfer(0)); with the
    raw x_ap that step would pull Pbar below rest and potentiate every active synapse during inhibition. Without
    inhibition this equals the rule with P = x_ap at lr_bas / 0.9. The literature rule adds a somatic-rate term
    (critical frequency for Ca2+ spike ignition); hp['burst_kappa'] (default 0) puts it back as
    P = apical_norm(x_ap) + kappa * x_som. Pbar is a per-neuron running average of P with time constant
    hp['burst_tau'] given in trials (converted to time steps); it is updated at every time step including the
    outcome step. It starts at its fixed point without apical activity: 0 plus kappa times the expected somatic rate
    per time step at gain 1 and the initial weights.

    Sign structure: synapses active while the apical dendrite is more active than usual are potentiated, synapses
    active while it is less active than usual (e.g. distractors at random times) are depressed. With kappa = 0 and
    the apical dendrite silent, P == Pbar == 0 and the rule is off; with kappa > 0 it stays active at gain 1 as
    kappa x_som (x_som - running mean of x_som).
    """
    p = apical_norm(x_ap) + hp.get('burst_kappa', 0.) * x_som
    if 'p_bar' not in state:
        state['p_bar'] = hp.get('burst_kappa', 0.) * state['_e_x_som_init'].expand_as(p).clone()
    state['p_bar'] += (p - state['p_bar']) / (hp['burst_tau'] * hp['_steps_per_trial'])
    return _pre_post(x_in, x_som * (p - state['p_bar']), hp)


# -------------------------------------------------------------------------------------------------------------------
# 2. BCM rule with sliding threshold (Bienenstock, Cooper & Munro 1982)
# -------------------------------------------------------------------------------------------------------------------

def bcm(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp, state):
    """
    dw = lr x_in x_som (x_som - theta_M),   tau dtheta_M/dt = x_som^2 / E0 - theta_M.

    Purely post-synaptic (no explicit apical factor; the apical gain enters through x_som). The sliding
    threshold is initialized at the fixed threshold theta of the mixed model and E0 is chosen per neuron so that
    its fixed point at initialization (gain 1, initial w_bas) is exactly theta: E0 = E_init[x_som^2] / theta. So at
    the start bcm equals hebb with an extra factor x_som, and as the apical gain raises x_som the threshold slides
    up (metaplasticity), making the rule self-limiting. hp['bcm_e0_scale'] multiplies E0 (> 1 lowers the
    threshold), hp['bcm_tau'] is the time constant in trials (inf = fixed threshold).
    """
    if 'theta_m' not in state:
        state['theta_m'] = thetas.clone()
        state['e0'] = hp['bcm_e0_scale'] * state['_e_x_som_sq_init'] / thetas
    tau_steps = hp['bcm_tau'] * hp['_steps_per_trial']
    state['theta_m'] += (x_som.pow(2) / state['e0'] - state['theta_m']) / tau_steps
    return _pre_post(x_in, x_som * (x_som - state['theta_m']), hp)


# -------------------------------------------------------------------------------------------------------------------
# 3. Urbanczik-Senn rule (Neuron 2014): dendritic prediction of the somatic rate
# -------------------------------------------------------------------------------------------------------------------

def us(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp, state):
    """
    dw = lr x_in (x_som - g_bar x_bas) = lr x_in x_bas (gain - g_bar).

    The basal compartment predicts the somatic rate from its own drive with the gain it usually experiences, g_bar;
    the apical input is the teacher that pulls the soma away from that prediction. g_bar is a per-neuron running
    average of the gain with time constant hp['us_tau'] in trials (updated at every time step including the outcome
    step), starting at the resting gain 1. With us_tau = inf the prediction stays at the resting gain 1, the rule is
    purely potentiating (the gain is never below 1) and it is self-limiting only through the apical signal itself
    (once the TD errors vanish the apical drive returns to rest). With a finite us_tau the rule also depresses
    synapses active while the apical gain is below its usual level, as in the original rule where the teacher can
    push both ways.
    """
    if 'g_bar' not in state:
        state['g_bar'] = torch.ones_like(gains)  # starts at the resting gain 1 (no apical activity)
    state['g_bar'] += (gains - state['g_bar']) / (hp['us_tau'] * hp['_steps_per_trial'])
    return _pre_post(x_in, x_bas * (gains - state['g_bar']), hp)


# -------------------------------------------------------------------------------------------------------------------
# 3b. Apical-deviation rule (user, 2026-10-19)
# -------------------------------------------------------------------------------------------------------------------

def apdev(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp, state):
    """
    dw = lr x_in (s - s_bar).

    The update of every active synapse is set by the neuron's apical signal s relative to its low-pass s_bar alone,
    with no post-synaptic basal factor: synapses active while the apical signal is above its usual level (the cue and
    touch on neurons with apical weight) are potentiated, synapses active while it is below (distractors at the other
    time steps once the outcome step has lifted s_bar) are depressed, by the same amount for every synapse active at
    that step. s_bar is a per-neuron running average with time constant hp['apdev_tau'] in trials (converted to time
    steps; inf keeps it at rest, so the rule only potentiates), updated at every time step including the outcome step
    and starting at rest. hp['apdev_signal'] = 'gain' uses s = gain (rest 1); 'x_ap' uses the apical activation itself
    (rest apical_transfer(0) = 1 / MAX_GAIN). The gain is MAX_GAIN x_ap for x_ap >= 1 / MAX_GAIN, so the two are the
    same rule (x_ap at lr * MAX_GAIN equals gain at lr) except at the inhibited outcome step, where x_ap = 0 pulls its
    low-pass below rest and every active synapse is then potentiated during apical inhibition, while the gain stays 1
    there as at rest and the rule stays silent. hp['apdev_bar_outcome'] = False is a causal ablation only: the low-pass
    skips the outcome time step (the last call of each trial), so the outcome-time apical activity no longer lifts it.
    hp['apdev_mult'] = True makes the update proportional to the weight, dw = lr x_in w (s - s_bar): synapses active
    together grow or shrink by the same factor, so the ratio of a neuron's two texture weights is kept (with an
    additive update both textures gain the same amount, because the apical signal at touch is the same in go and no-go
    trials, and the neuron loses its go/no-go contrast). hp['apdev_bar'] = False drops the low-pass, dw = lr x_in s:
    every active synapse is potentiated, at rest too (by lr per step with the gain), so only a normalizing bound (l1,
    pre, both) makes the inputs compete; apdev_tau = inf instead subtracts the rest value, dw = lr x_in (s - rest),
    which is silent at rest. Neither has a depression term. hp['apdev_mult'] = 'bas' multiplies the update by the
    neuron's un-gained basal drive instead of the weight, dw = lr x_in x_bas (s - s_bar) (pre, basal post and the
    apical deviation: a three-factor Hebbian form). 'wbas' multiplies by both, dw = lr x_in w x_bas (s - s_bar): the
    increment per unit weight is then the neuron's drive, so of the synapses active at a step the one that drives the
    neuron most grows fastest, and the background and coincident inputs in x_bas no longer lift the weak ones.
    hp['apdev_bas_power'] raises x_bas to that power in 'bas' and 'wbas'. hp['apdev_xbar'] replaces the presynaptic
    factor x_in by x_in - x_bar (user, 2026-10-21), x_bar a running average of each input over hp['apdev_xbar_tau']
    trials, used before it is updated and starting at 0; it is presynaptic, so the same for every neuron, and synapses
    of inputs absent at a step are then depressed by x_bar. 'step' averages over all time steps (the input's frequency
    per step, below 1 / 17 for every input here); 'slot' averages each time step of the trial over trials (the input
    expected at that moment: 1 for the tone at its step, 0.5 for each texture at the touch, p / 17 for a distractor at
    any step), so the tone is never potentiated and at each touch the absent texture loses what the present one gains.
    'apical' is local instead: each synapse averages its input over the stimulus steps weighted by its neuron's apical
    activation apical_norm(x_ap) (the input expected during the neuron's apical events; tau then counts full-gain
    events, and the outcome step, which has no basal input, is left out of the average): with equal gains at cue and
    touch it is about 1/2 for the tone and 1/4 for each texture, so the tone's cue gain and touch loss about cancel.
    """
    use_gain = hp.get('apdev_signal', 'gain') == 'gain'
    sig = gains if use_gain else x_ap
    if 's_bar' not in state:
        state['s_bar'] = torch.full_like(sig, 1. if use_gain else A_REST)  # starts at rest
    n_call = state.get('_apdev_calls', 0)
    state['_apdev_calls'] = n_call + 1
    outcome_step = n_call % hp['_steps_per_trial'] == hp['_steps_per_trial'] - 1
    if hp.get('apdev_bar_outcome', True) or not outcome_step:
        state['s_bar'] += (sig - state['s_bar']) / (hp['apdev_tau'] * hp['_steps_per_trial'])
    pre = x_in
    xbar = hp.get('apdev_xbar')
    if xbar:
        if 'x_bar' not in state:
            shape = {'step': x_in.shape, 'slot': (x_in.shape[0], hp['_steps_per_trial']),
                     'apical': w_bas.shape}[xbar]
            state['x_bar'] = torch.zeros(shape)
        if xbar == 'step':
            pre = x_in - state['x_bar']
            state['x_bar'] += (x_in - state['x_bar']) / (hp['apdev_xbar_tau'] * hp['_steps_per_trial'])
        elif xbar == 'slot':
            t = n_call % hp['_steps_per_trial']
            pre = x_in - state['x_bar'][:, t]
            state['x_bar'][:, t] += (x_in - state['x_bar'][:, t]) / hp['apdev_xbar_tau']
        else:  # 'apical', per synapse
            pre = x_in[:, None] - state['x_bar']
            if not outcome_step:
                state['x_bar'] += pre * apical_norm(x_ap)[None, :] / hp['apdev_xbar_tau']
    post = sig - state['s_bar'] if hp.get('apdev_bar', True) else sig
    dw = hp['lr_bas'] * pre * post[None, :] if pre.dim() == 2 else _pre_post(pre, post, hp)
    mult = hp.get('apdev_mult', False)
    if mult in ('bas', 'wbas'):
        power = hp.get('apdev_bas_power', 1)
        dw = dw * (x_bas if power == 1 else x_bas ** power)
        return dw * w_bas if mult == 'wbas' else dw
    return dw * w_bas if mult else dw


# -------------------------------------------------------------------------------------------------------------------
# 4. Calcium-threshold rule (Lisman 1989; Shouval, Bear & Cooper 2002; Graupner & Brunel 2012)
# -------------------------------------------------------------------------------------------------------------------

def calcium(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp, state):
    """
    Per-synapse calcium proxy with an NMDA term (pre- and post-synaptic activity) and a dendritic Ca2+-spike term
    (post-synaptic activity times apical activation), both requiring the presynaptic input x_in:

        c = x_in x_som (1 + ca_beta * apical_norm(x_ap))

    Two thresholds on the calcium level: LTD between theta_d and theta_p, LTP above theta_p (smooth sigmoids):

        dw = lr x_in [ gamma_p sigmoid((c - theta_p) / s) - gamma_d sigmoid((c - theta_d) / s) ]

    The LTP threshold is set per neuron relative to the strongest basal response the neuron can produce without
    apical input, theta_p = hp['ca_theta_p'] * max_i w_bas_i(init), so with ca_theta_p > 1 LTP needs the apical
    gain (the model's theta is 4x the expected drive per time step, which is far below any driven response because
    stimuli are sparse in time, so a threshold at theta would make LTP independent of the gain). theta_d = hp['ca_ltd_frac'] *
    theta_p: with ca_ltd_frac below the ratio of the un-gained response to theta_p, apical-free activity of the
    dominant synapse falls in the LTD window and is depressed; closer to 1 the rule is silent without apical input.
    The smoothness is s = hp['ca_smooth'] * theta_p, gamma_p = 1 and gamma_d = hp['ca_gamma_d']. Because x_in is
    binary, c is the same for all synapses active on a neuron, so the post-synaptic factor is per neuron. The update
    magnitude is bounded by lr per step independently of how far c is above threshold (unlike hebb).
    """
    if 'theta_p' not in state:
        state['theta_p'] = hp['ca_theta_p'] * state['_w_max_init']
    theta_p = state['theta_p']
    c = x_som * (1 + hp['ca_beta'] * apical_norm(x_ap))
    s = hp['ca_smooth'] * theta_p
    pot = torch.sigmoid((c - theta_p) / s)
    dep = hp['ca_gamma_d'] * torch.sigmoid((c - hp['ca_ltd_frac'] * theta_p) / s)
    return _pre_post(x_in, pot - dep, hp)


# -------------------------------------------------------------------------------------------------------------------

def apply_bound(w_bas: torch.Tensor, dw, hp: dict) -> torch.Tensor:
    """Apply the update dw to w_bas with the bounding mechanism hp['bound'] (see module docstring)."""
    if dw is None:
        return w_bas
    bound = hp['bound']
    if bound == 'soft':
        return torch.clamp(w_bas + dw * (1 - w_bas) * w_bas, min=0, max=1)
    if bound == 'clamp':
        return torch.clamp(w_bas + dw, min=0, max=1)
    if bound == 'l1':
        w = torch.clamp(w_bas + dw, min=0)
        return torch.clamp(w * (hp['_col_sums'] / torch.clamp(w.sum(0, keepdim=True), min=1e-12)), max=1)
    if bound == 'none':
        return w_bas + dw
    if bound in ('pre', 'both'):
        w = torch.clamp(w_bas + dw, min=0)
        if bound == 'both':
            w = w * (hp['_col_sums'] / torch.clamp(w.sum(0, keepdim=True), min=1e-12))
        row = w.sum(1, keepdim=True)
        return torch.clamp(w * (hp['_row_sums'] / torch.clamp(row, min=1e-12)), max=1)
    raise ValueError(f"unknown bound {bound!r}")


RULES = {
    'none': none,
    'hebb': hebb,
    'gated_hebb': gated_hebb,
    'oja': oja,
    'burst': burst,
    'bcm': bcm,
    'us': us,
    'apdev': apdev,
    'calcium': calcium,
}

# Rule-specific hyperparameters and their defaults (merged into DEFAULT_HP by simulate.py)
RULE_HP = dict(
    gate_power=1.0,     # gated_hebb
    gh_post='som',      # gated_hebb: post-synaptic variable compared with the threshold ('som' or 'bas')
    gh_theta_scale=1.0, # gated_hebb: multiplies the threshold in the basal rule only
    oja_decay=1.0,      # oja
    burst_tau=20.,      # burst: running-average time constant of the burst fraction, in trials
    burst_kappa=0.,     # burst: somatic-rate contribution to the burst fraction
    bcm_tau=100.,       # bcm: sliding-threshold time constant, in trials (inf = fixed threshold)
    us_tau=float('inf'),  # us: time constant (trials) of the running-average gain used as prediction (inf = resting gain)
    apdev_tau=20.,      # apdev: time constant (trials) of the low-pass of the apical signal (inf = rest)
    apdev_signal='gain',  # apdev: apical signal, 'gain' or 'x_ap' (the same rule at lr * MAX_GAIN, see apdev)
    apdev_bar_outcome=True,  # apdev: False (causal ablation only) keeps the outcome step out of the low-pass
    apdev_mult=False,   # apdev: True multiplies the update by the weight, 'bas' by the neuron's un-gained basal drive,
                        # 'wbas' by both
    apdev_bas_power=1,  # apdev: power of x_bas in apdev_mult 'bas' and 'wbas'
    apdev_xbar=None,    # apdev: presynaptic factor x_in - x_bar, x_bar averaged over all steps ('step'), per step of
                        # the trial over trials ('slot') or per synapse over the neuron's apical events ('apical');
                        # None keeps x_in
    apdev_xbar_tau=50., # apdev: time constant (trials) of x_bar
    apdev_bar=True,     # apdev: False drops the low-pass, dw = lr x_in s (potentiation only, at rest too)
    bcm_e0_scale=1.0,   # bcm: scales E0 (> 1 lowers the threshold fixed point)
    ca_beta=2.0,        # calcium: weight of the apical Ca2+-spike term relative to the NMDA term
    ca_theta_p=1.5,     # calcium: LTP threshold as a multiple of the neuron's strongest initial basal weight
    ca_ltd_frac=0.5,    # calcium: LTD threshold as a fraction of the LTP threshold
    ca_gamma_d=0.5,     # calcium: depression rate relative to potentiation rate
    ca_smooth=0.15,     # calcium: threshold smoothness as a fraction of the LTP threshold
)
