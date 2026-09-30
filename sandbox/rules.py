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
    'l1'     w <- normalize(clamp(w + dw, 0), L1, dim=0)  each neuron's afferents keep summing to 1 (competition)
    'none'   w <- w + dw                                 nothing (only sensible for rules with their own decay)
    'pre'    w <- rescale rows of clamp(w + dw, 0)       each stimulus keeps its initial total outgoing weight (sum
                                                         over neurons), so a rule can redistribute a stimulus's drive
                                                         across neurons but never grow it (presynaptic competition)
    'both'   per-neuron L1 (as 'l1') followed by 'pre'   both constraints, one step of each per update

The rules are called once per time step of the stimulus period and once at the outcome time step (with x_in = 0),
so slow variables also see the outcome-time apical activity.

Mapping of the literature rules in sandbox/plasticity_rules_tmp.py onto this model
-----------------------------------------------------------------------------------
The neuron there is  x_som = gain(x_ap) * x_bas  with x_bas = w_bas . x_pre, which is exactly this model with
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
    bursting, which is what raises the somatic gain. So P = x_ap, no separate sigmoid needed. The literature rule
    adds a somatic-rate term (critical frequency for Ca2+ spike ignition); hp['burst_kappa'] (default 0) puts it back
    as P = x_ap + kappa * x_som. Pbar is a per-neuron running average of P with time constant hp['burst_tau']
    given in trials (converted to time steps); it is updated at every time step including the outcome step.

    Sign structure: synapses active while the apical dendrite is more active than usual are potentiated, synapses
    active while it is less active than usual (e.g. distractors at random times) are depressed. With the apical
    dendrite silenced P == Pbar and the rule is off, so it cannot support learning under apical inhibition.
    """
    p = x_ap + hp.get('burst_kappa', 0.) * x_som
    if 'p_bar' not in state:
        state['p_bar'] = p.clone()
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

    The basal compartment predicts the somatic rate from its own drive with the gain it usually experiences,
    g_bar; the apical input is the teacher that pulls the soma away from that prediction. g_bar is a per-neuron
    running average of the gain with time constant hp['us_tau'] in trials (updated at every time step including the
    outcome step). With us_tau = inf the prediction stays at the resting gain 1, the rule is purely potentiating
    (the gain is never below 1) and it is self-limiting only through the apical signal itself (once the TD errors
    vanish the apical drive returns to rest). With a finite us_tau the rule also depresses synapses active while the
    apical gain is below its usual level, as in the original rule where the teacher can push both ways.
    """
    if 'g_bar' not in state:
        state['g_bar'] = gains.clone()
    state['g_bar'] += (gains - state['g_bar']) / (hp['us_tau'] * hp['_steps_per_trial'])
    return _pre_post(x_in, x_bas * (gains - state['g_bar']), hp)


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
        return torch.nn.functional.normalize(torch.clamp(w_bas + dw, min=0), p=1., dim=0)
    if bound == 'none':
        return w_bas + dw
    if bound in ('pre', 'both'):
        w = torch.clamp(w_bas + dw, min=0)
        if bound == 'both':
            w = torch.nn.functional.normalize(w, p=1., dim=0)
        row = w.sum(1, keepdim=True)
        return w * (hp['_row_sums'] / torch.clamp(row, min=1e-12))
    raise ValueError(f"unknown bound {bound!r}")


RULES = {
    'none': none,
    'hebb': hebb,
    'gated_hebb': gated_hebb,
    'oja': oja,
    'burst': burst,
    'bcm': bcm,
    'us': us,
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
    bcm_e0_scale=1.0,   # bcm: scales E0 (> 1 lowers the threshold fixed point)
    ca_beta=2.0,        # calcium: weight of the apical Ca2+-spike term relative to the NMDA term
    ca_theta_p=1.5,     # calcium: LTP threshold as a multiple of the neuron's strongest initial basal weight
    ca_ltd_frac=0.5,    # calcium: LTD threshold as a fraction of the LTP threshold
    ca_gamma_d=0.5,     # calcium: depression rate relative to potentiation rate
    ca_smooth=0.15,     # calcium: threshold smoothness as a fraction of the LTP threshold
)
