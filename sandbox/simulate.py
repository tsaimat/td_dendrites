"""
Copy of l5apical.simulations.simulate_seed with (1) a pluggable bottom-up plasticity rule, (2) hyperparameters
passed in as a dict and (3) periodic snapshots of the basal weights. Kept as a verbatim mirror of the original
trial loop except at the lines marked "SANDBOX", so that sandbox/check_reproduces.py can assert bit-identity
with the main code. If the main loop changes, re-copy it here.
"""
from random import seed as rd_seed
from random import shuffle as random_shuffle
import numpy as np
import torch
from l5apical.helper import *
from l5apical.simulations import Agent
from sandbox.rules import RULES, RULE_HP, apply_bound

# Default hyperparameters. Everything a sweep may vary lives here.
DEFAULT_HP = dict(
    rule='hebb',              # key into sandbox.rules.RULES ('hebb' + 'soft' is the rule in the main code)
    bound='soft',             # how the update is kept bounded, see sandbox.rules.apply_bound
    lr_bas=LR,                # learning rate of the basal weights (main code uses lr_ap = LR)
    lr_ap=LR,                 # learning rate of the apical weights (main code: LR)
    lr_policy=LR,             # learning rate of the agent's policy network (main code: LR)
    lr_td=LR,                 # learning rate of the TD value estimator (main code: LR)
    lr_trace=LR,              # learning rate of the apical afferent trace x_pre_t (main code: LR)
    theta_0=THETA_0,          # scale of the per-neuron plasticity thresholds (as in the mixed model)
    theta_bas_0=None,         # threshold scale seen by the basal rule only (None: same thresholds as the apical rule)
    n_noise=N_Z_BU_PLASTICITY,  # number of distractor stimuli
    n_z=N_Z,                  # number of pyramidal neurons
    init='random',            # 'random' (L1-normalized rand**init_power, as in the mixed model) or 'identity'
    init_power=6,             # exponent applied to the uniform draws before normalization (sparsity of the init)
    n_trials=N_TRIALS,
    n_trials_inhibited=0,     # apical dendrites silenced for the first this many trials (main code: N_TRIALS of N_TRIALS_AP_INH)
    snapshot_every=50,        # store w_bas every this many trials (plus the final one)
    store_full=False,         # keep every per-trial array of the main code (large); False drops the ones no sandbox
                              # metric or plot uses and stores the rest as float32
    **RULE_HP,                # rule-specific hyperparameters, see sandbox.rules
)
K_W_BAS_H = 'w_bas_history'   # extra result key: (n_snapshots, n_stim, n_z)
K_W_BAS_T = 'w_bas_history_t'  # trial index of each snapshot
K_HP = 'hp'
K_DIVERGED = 'diverged_at'  # trial at which the simulation produced non-finite values (None if it did not)
K_POLICY_H = 'policy_w_history'  # policy weights of the agent at the w_bas snapshot trials
K_STATE = 'rule_state'      # final slow variables of the rule (numpy)
K_STATE_H = 'rule_state_history'  # per-trial mean over neurons of every slow variable


def get_params_bu(hp: dict):
    """
    Mirror of l5apical.simulations.get_params for the bottom-up model, consuming the random number generators in
    the same order so that the default hyperparameters reproduce Simulation.BU_PLASTICITY exactly.
    """
    n_task_stim = 3
    n_noise_stim = int(hp['n_noise'])
    n_stim = n_task_stim + n_noise_stim
    n_z = int(hp['n_z'])
    weights_apical = 0.05 + torch.rand(n_z) * 0.1
    noise_probs = [float(i / (n_noise_stim + 1)) for i in range(1, n_noise_stim + 1)]
    random_shuffle(noise_probs)
    if hp['init'] == 'random':
        w_bas = torch.nn.functional.normalize(torch.rand((n_stim, n_z)) ** hp['init_power'], p=1., dim=0)
        probs = torch.tensor([1, 0.5, 0.5] + noise_probs) / N_TIME_STEPS
        thetas = hp['theta_0'] * torch.matmul(probs[:], w_bas) + BKG
    elif hp['init'] == 'identity':
        if n_stim != n_z:
            raise ValueError(f"identity init needs n_noise = n_z - {n_task_stim} = {n_z - n_task_stim}, got {n_noise_stim}")
        w_bas = torch.diag(torch.ones(n_z))
        thetas = THETA * torch.ones(n_z) + BKG
    else:
        raise ValueError(hp['init'])
    return n_stim, n_noise_stim, n_z, noise_probs, LR, w_bas, thetas.float(), weights_apical, int(hp['n_trials'])


def simulate_seed_bu(hp: dict = None, seed: int = 0) -> dict:
    """
    Simulate one seed of the go/no-go task with a pluggable bottom-up plasticity rule.
    :param hp: hyperparameter dict; missing keys are filled from DEFAULT_HP
    :param seed: pseudorandom seed
    :return: result dict with the same keys as l5apical plus the w_bas snapshots and the hyperparameters
    """
    hp = {**DEFAULT_HP, **(hp or {})}
    hp['_steps_per_trial'] = N_TIME_STEPS
    rule = RULES[hp['rule']]
    state = {}
    n_inhibited = int(hp['n_trials_inhibited'])

    # Setting random seed
    rd_seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    # Initialization of basic params
    apical_active = n_inhibited == 0  # SANDBOX
    n_inputs, noise_inputs, n_z, noise_ps, lr, w_bas, thetas, w_ap, n_trials = get_params_bu(hp)  # SANDBOX
    # SANDBOX: expected x_som^2 per time step at initialization (gain 1), used by the bcm rule to set its threshold
    stim_probs = torch.tensor([1, 0.5, 0.5] + noise_ps) / N_TIME_STEPS
    state['_e_x_som_sq_init'] = (torch.matmul(stim_probs, (w_bas + BKG) ** 2)
                                 + torch.clamp(1 - stim_probs.sum(), min=0) * BKG ** 2)
    state['_w_max_init'] = w_bas.max(0).values + BKG  # SANDBOX: strongest un-gained response, used by calcium
    # SANDBOX: the basal rule may use its own threshold scale; the apical rule always keeps thetas
    thetas_bas = thetas if hp.get('theta_bas_0') is None else (
        float(hp['theta_bas_0']) * torch.matmul(stim_probs, w_bas) + BKG).float()
    # SANDBOX: the four learning rates the main code ties to LR are separate hyperparameters here
    tlr, learning_rate = float(hp['lr_trace']), float(hp['lr_td'])
    lr_ap, plr = float(hp['lr_ap']), float(hp['lr_policy'])

    # Relevant simulation time steps (first simulated time step starts with the tone cue)
    tone_t = 0
    texture_t = TEXTURE_T - TONE_T
    outcome_t = OUTCOME_T - TONE_T
    n_timings = N_TIME_STEPS

    # Basal input initializations
    tx_idxs = torch.tensor([T1_IDX, T2_IDX])
    texture_xs = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    base_input_time_idxs = np.zeros(shape=n_inputs)
    base_input_time_idxs[0] = tone_t
    base_input_time_idxs[1] = texture_t
    base_input_time_idxs[2] = texture_t
    x_bas_background = torch.ones(n_z) * BKG

    # Initialize agent and state value estimator
    actor = Agent(nr_inputs=n_z, policy_lr=plr)
    x_pre_t = torch.zeros(outcome_t)
    dw_weights = torch.zeros(n_z)

    # Initialization of empty objects in which various simulation variables will be recorded
    outcomes, correct_trials, den_out = tuple(nan_array((n_trials,)) for _ in range(3))
    d_sen_tx, d_sen_t1, d_sen_t2, x_pre_t_h, td_delta_h = tuple(nan_array((n_trials, n_timings)) for _ in range(5))
    w_ap_h, gain_h, x_som_texture_h = tuple(nan_array((n_trials, n_z)) for _ in range(3))
    z_h = np.zeros((n_trials, n_z))
    v_hat_h = np.zeros((n_trials, 1 + n_timings))
    snap = int(hp['snapshot_every'])  # SANDBOX
    snap_trials = sorted(set(list(range(0, n_trials, snap)) + [n_trials - 1]))  # SANDBOX
    w_bas_h = np.zeros((len(snap_trials), n_inputs, n_z), dtype=np.float32)  # SANDBOX
    policy_h = np.zeros((len(snap_trials), n_z), dtype=np.float32)  # SANDBOX
    state_h = {}  # SANDBOX

    # Simulating trial after trial
    diverged_at = None  # SANDBOX
    for j in range(n_trials):

        # Lift apical inhibition (in case it was present)
        if j == n_inhibited and n_inhibited > 0:  # SANDBOX
            apical_active = True

        # SANDBOX: a diverging rule makes the lick probability NaN; stop and return what was recorded so far
        if not (torch.isfinite(w_bas).all() and torch.isfinite(dw_weights).all() and torch.isfinite(x_pre_t).all()
                and torch.isfinite(actor.policy_weights).all()):
            diverged_at = j
            break

        # Initialize trial
        texture = np.random.choice([0, 1])
        txt_in = texture_xs[texture]
        noise_in = torch.tensor([np.random.choice(a=[0, 1], p=[1-pr, pr]) for pr in noise_ps])
        basal_input = torch.cat(tensors=(txt_in, noise_in), dim=0)
        input_time_idxs = torch.tensor(np.copy(base_input_time_idxs))
        input_time_idxs[3:] = torch.tensor(np.random.randint(low=0, high=outcome_t, size=noise_inputs))
        input_time_idxs[:][basal_input < 0.5] = n_timings

        # Storage variables
        dv_hat_before = torch.tensor([0.])
        v_hat_h[j, 0] = dv_hat_before.numpy()
        d_sen_tx[j, :outcome_t] = torch.mean(apical_transfer(torch.outer(x_pre_t, w_ap[tx_idxs])), dim=1)
        d_sen_t1[j, :outcome_t] = apical_transfer(x_pre_t * w_ap[T1_IDX])
        d_sen_t2[j, :outcome_t] = apical_transfer(x_pre_t * w_ap[T2_IDX])

        # Simulate each time step of the trial up to right before the action and outcome happen
        for t in range(outcome_t):

            # Simulate pyramidal neurons and compute state value estimate
            current_idxs = input_time_idxs == t
            x_in = torch.zeros((n_inputs,))
            x_in[current_idxs] = 1
            x_bas = torch.matmul(x_in, w_bas) + x_bas_background
            x_ap = apical_transfer(w_ap * x_pre_t[t])  # SANDBOX: split in two lines to hand x_ap to the rule
            gains = gain(x_ap)
            x_som = gains * x_bas
            actor.integrate_input(x_som)
            dv_hat_now = torch.dot(dw_weights, x_som)

            # Update state value estimator, sensory dendrite afferent and sensory dendrite synaptic weight
            dw_weights += learning_rate * dv_hat_now * z_h[j, :] * K_V
            w_ap = torch.clamp(w_ap + lr_ap * x_pre_t[t] * (x_som - thetas) * (1 - w_ap) * w_ap, min=0, max=1)
            w_bas = apply_bound(w_bas, rule(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas_bas, hp, state), hp)  # SANDBOX
            x_pre_t[t] += tlr * (abs(dv_hat_now.item()) - x_pre_t[t])

            # Store variables of interest
            z_h[j, :] = GAMMA * z_h[j, :] + x_som.numpy()
            v_hat_h[j, 1 + t] = v_hat_h[j, t] / GAMMA + dv_hat_now.numpy()
            td_delta_h[j, t] = GAMMA * dv_hat_now
            driven = (x_bas - x_bas_background) > 0
            gain_h[j, driven.numpy()] = gains[driven].detach().clone().numpy()
            if t == texture_t:
                x_som_texture_h[j, :] = x_som.detach().clone().numpy()

        # Select the action, observe the reward obtained and update the policy network accordingly
        reward, licked, correct_trials[j] = actor.act(texture)

        # Record the outcome and the predicted reward
        if licked:
            r_pred = v_hat_h[j, outcome_t]
            outcomes[j] = Outcome.HIT.value if texture else Outcome.FA.value
        else:
            r_pred = 0.
            outcomes[j] = Outcome.MISS.value if texture else Outcome.CR.value

        # Outcome timing apical dendrite activities
        out_salience = torch.abs(reward - r_pred).clone().detach()
        sen_salience = abs(r_pred) * w_ap
        if apical_active:
            x_ap = apical_transfer(sen_salience + out_salience)
            den_out[j] = apical_transfer(out_salience)
            d_sen_tx[j, outcome_t] = torch.mean(apical_transfer(sen_salience[tx_idxs]))
            d_sen_t1[j, outcome_t] = apical_transfer(sen_salience[T1_IDX])
            d_sen_t2[j, outcome_t] = apical_transfer(sen_salience[T2_IDX])
        else:
            x_ap = torch.tensor([0.])
            den_out[j] = 0.
            d_sen_tx[j, outcome_t] = 0.
            d_sen_t1[j, outcome_t] = 0.
            d_sen_t2[j, outcome_t] = 0.

        # Update state-value estimate and policy network according to outcome activity
        gains = gain(x_ap)
        x_som = gains * x_bas_background
        # SANDBOX: let the rule's slow variables see the outcome-time apical activity (no presynaptic input, so
        # every rule returns dw = 0 here)
        w_bas = apply_bound(w_bas, rule(w_bas, torch.zeros(n_inputs), x_som, x_bas_background,
                                        x_ap.expand(n_z), gains.expand(n_z), thetas_bas, hp, state), hp)
        dw_weights += learning_rate * (reward - r_pred) * z_h[j, :] * (x_som - BKG)
        actor.update(reward=reward.item(), x_som=x_som)

        # Record variables of interest for post-simulation analysis
        v_hat_h[j, outcome_t+1] = v_hat_h[j, outcome_t] - r_pred
        w_ap_h[j, :] = w_ap
        x_pre_t_h[j, :-1] = x_pre_t
        x_pre_t_h[j, outcome_t] = abs(r_pred)
        td_delta_h[j, outcome_t] = reward - r_pred
        if j in snap_trials:  # SANDBOX
            w_bas_h[snap_trials.index(j)] = w_bas.numpy()
            policy_h[snap_trials.index(j)] = actor.policy_weights.numpy()
        for k, v in state.items():  # SANDBOX
            if not k.startswith('_'):
                state_h.setdefault(k, np.zeros(n_trials))[j] = float(v.mean())

        # Update weights according to TD error elicited by resetting value estimate to zero at the end of the trial
        dw_weights += - learning_rate * torch.tensor(v_hat_h[j, outcome_t+1] * z_h[j, :]).float() * K_V

    z_h[z_h == 0] = np.nan
    if not hp['store_full']:  # SANDBOX: slim results (a 4000-trial seed is ~50 MB otherwise)
        big = (z_h, gain_h, td_delta_h, d_sen_tx, d_sen_t1, d_sen_t2, x_som_texture_h)
        z_h, gain_h, td_delta_h, d_sen_tx, d_sen_t1, d_sen_t2, x_som_texture_h = (np.zeros(0) for _ in big)
        v_hat_h, w_ap_h, x_pre_t_h = (a.astype(np.float32) for a in (v_hat_h, w_ap_h, x_pre_t_h))
    if diverged_at is not None:  # SANDBOX: keep the snapshots taken so far, blank the rest
        w_bas_h[[i for i, t in enumerate(snap_trials) if t >= diverged_at]] = np.nan
        w_bas_h[-1] = w_bas.numpy()
        for v in state_h.values():
            v[diverged_at:] = np.nan
    return {K_OUTCOME: outcomes,
            K_CORRECT_TRIALS: correct_trials,
            K_V_HAT: v_hat_h,
            K_TD_DELTA: td_delta_h,
            K_DENDRITE_SENSORY: d_sen_tx,
            K_DENDRITE_SENSORY_T1: d_sen_t1,
            K_DENDRITE_SENSORY_T2: d_sen_t2,
            K_DENDRITE_OUTCOME: den_out,
            K_Z: z_h,
            K_GAIN: gain_h,
            K_W_AP: w_ap_h,
            K_TIMINGS: x_pre_t_h,
            K_W_BAS: w_bas,
            K_X_SOM_TXT: x_som_texture_h,
            K_W_BAS_H: w_bas_h,
            K_W_BAS_T: np.array(snap_trials),
            K_POLICY_H: policy_h,
            K_STATE: {k: v.numpy() for k, v in state.items() if not k.startswith('_')},
            K_STATE_H: state_h,
            K_DIVERGED: diverged_at,
            K_HP: hp}


# ---------------------------------------------------------------------------------------------------------------
# Cheap Python-side metrics, so sweeps can be compared without the Matlab performance estimator
# ---------------------------------------------------------------------------------------------------------------

def moving_performance(correct_trials: np.ndarray, window: int = 100) -> np.ndarray:
    """Trailing moving average of the fraction of correct trials (NaN for the first window-1 trials)."""
    c = np.asarray(correct_trials, dtype=float)
    out = np.full_like(c, np.nan)
    cs = np.concatenate([[0.], np.cumsum(c)])
    out[window - 1:] = (cs[window:] - cs[:-window]) / window
    return out


def expert_trial_proxy(correct_trials: np.ndarray, threshold: float = 0.8, window: int = 100) -> float:
    """
    Proxy for the Smith et al. expert trial: the trial after which the moving performance never drops below the
    threshold again. NaN if the agent never reaches the threshold or is not above it at the end.
    """
    perf = moving_performance(correct_trials, window)
    if np.isnan(perf[-1]) or perf[-1] < threshold:  # never expert, or diverged (NaN trials at the end)
        return np.nan
    below = np.where(perf < threshold)[0]
    return float(below[-1] + 1) if len(below) else float(window - 1)


def task_weight_fraction(w_bas: np.ndarray) -> np.ndarray:
    """Per neuron: fraction of total basal weight that comes from the three task stimuli (cue, T1, T2)."""
    w = np.asarray(w_bas, dtype=float)
    return w[:3].sum(0) / np.maximum(w.sum(0), 1e-12)


def texture_selectivity(w_bas: np.ndarray) -> np.ndarray:
    """Per neuron: |w_T1 - w_T2| / (w_T1 + w_T2), 0 = equal texture drive, 1 = purely texture selective."""
    w = np.asarray(w_bas, dtype=float)
    return np.abs(w[T1_IDX] - w[T2_IDX]) / np.maximum(w[T1_IDX] + w[T2_IDX], 1e-12)


def detector_counts(w_bas: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Number of neurons whose strongest basal weight exceeds the threshold, split by what that input is: the tone cue,
    one of the textures, or a distractor. 'Texture detectors' are what the apical rule can turn into task neurons.
    """
    w = np.asarray(w_bas, dtype=float)
    dom, strong = w.argmax(0), w.max(0) > threshold
    return dict(det_tone=int(np.sum(strong & (dom == 0))), det_texture=int(np.sum(strong & ((dom == 1) | (dom == 2)))),
                det_distractor=int(np.sum(strong & (dom > 2))))


SUMMARY_KEYS = ['diverged', 'expert_proxy', 'expert_after_lift', 'perf_inhibited_end', 'final_perf', 'w_bas_change',
                'w_bas_change_inhibited', 'w_bas_sum_mean', 'w_bas_min', 'w_bas_max', 'task_frac_start', 'task_frac_end',
                'selectivity_start', 'selectivity_end', 'det_tone_start', 'det_texture_start', 'det_distractor_start',
                'det_tone', 'det_texture', 'det_distractor', 'w_ap_mean', 'w_ap_max']


def summarize(result: dict) -> dict:
    """Scalar summary of one seed for sweep tables."""
    w0, w1 = result[K_W_BAS_H][0], result[K_W_BAS_H][-1]
    correct = result[K_CORRECT_TRIALS]
    perf = moving_performance(correct)
    n_inh = int(result[K_HP]['n_trials_inhibited'])
    if n_inh > 0:  # metrics of the apical inhibition protocol: performance while inhibited, expert trial after lift
        expert_after_lift = expert_trial_proxy(correct[n_inh:])
        perf_inhibited_end = float(np.nanmean(correct[max(0, n_inh - 200):n_inh]))
        i_lift = int(np.searchsorted(result[K_W_BAS_T], n_inh))  # first snapshot at or after the lift
        w_bas_change_inhibited = float(np.abs(result[K_W_BAS_H][i_lift] - w0).mean())
    else:
        expert_after_lift, perf_inhibited_end, w_bas_change_inhibited = np.nan, np.nan, np.nan
    return dict(
        diverged=float(result[K_DIVERGED] is not None),
        expert_proxy=expert_trial_proxy(correct),
        expert_after_lift=expert_after_lift,
        perf_inhibited_end=perf_inhibited_end,
        final_perf=float(np.nanmean(perf[-200:])),
        w_bas_change=float(np.abs(w1 - w0).mean()),
        w_bas_change_inhibited=w_bas_change_inhibited,
        w_bas_sum_mean=float(w1.sum(0).mean()),
        w_bas_min=float(w1.min()), w_bas_max=float(w1.max()),
        task_frac_start=float(task_weight_fraction(w0).mean()),
        task_frac_end=float(task_weight_fraction(w1).mean()),
        selectivity_start=float(texture_selectivity(w0).mean()),
        selectivity_end=float(texture_selectivity(w1).mean()),
        **{k + '_start': v for k, v in detector_counts(w0).items()},
        **detector_counts(w1),
        w_ap_mean=float(result[K_W_AP][-1].mean()), w_ap_max=float(result[K_W_AP][-1].max()),
    )
