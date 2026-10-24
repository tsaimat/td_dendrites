"""
Helper functions and constants for the L5 apical dendrite model of sensory learning, including the estimator of the
learning curve of Smith et al. (2004) that extracts the performance traces
"""

import torch
import numpy as np
from typing import Literal
from enum import Enum
from pathlib import Path
from scipy import special, stats

ROOT_DIR = Path(__file__).parent.parent.resolve()  # Root directory

# Main simulation parameters
N_Z = 1000  # Number of pyramidal neurons
N_NOISE = 30  # Number of distractor stimuli
INIT_SIGMA = 2.  # Log-standard deviation of the lognormal initialization of the basal weights
INIT_SCALE = 1.89  # Total initial basal input per neuron (one third of the neurons respond to the textures)
INIT_CLIP = 1.  # Upper bound of the initial basal weights (all basal weights stay in [0, 1])
N_TRIALS = 1800  # Default number of trials simulated
N_TRIALS_AP_INH = 4000  # Number of trials in case the apical dendrites are inhibited for the first 1800 trials
MAX_GAIN = 10.  # Maximum possible apical gain
BKG = 0.01  # Background activity of pyramidal neurons
K_V = BKG * MAX_GAIN  # Constant for TD learning plasticity
REWARD = 1.  # Reward value in case of hit
PUNISHMENT = -0.5  # Reward value in case of false alarm
GAMMA = 1.  # TD learning gamma
N_SEEDS = 10  # Number of pseudorandom seeds simulated
HZ = 4  # Number of time steps simulated per second
TONE_T = HZ  # Time step when tone happens (1s in trial)
TEXTURE_T = 2 * HZ + 1  # Time step when texture is perceived (2.25s in trial)
OUTCOME_T = 5 * HZ + 1  # Time when outcome is perceived (5.25s in trial)
N_TIME_STEPS = 5 * HZ + 2 - TONE_T  # Total number of time steps per trial
TONE_IDX = 0  # Index of the tone stimulus
T1_IDX = 2  # Index of the texture 1 stimulus
T2_IDX = 1  # Index of the texture 2 stimulus
N_TASK_STIMULI = 3  # Number of task stimuli (the tone and the two textures), followed by the distractors
CA_SMOOTH = 0.15  # Smoothness of the thresholds of the calcium rule, as a fraction of its LTP threshold
SNAPSHOT_EVERY = 50  # Number of trials between two stored snapshots of the basal weights (bottom-up plasticity)

# Hyperparameters of the two learning regimes: the biological one produces all the figures of the manuscript except the
# supplementary figure of the fast-learning regime. theta_0 scales the thresholds of the apical plasticity rule, the
# learning rates are those of the policy network (lr_policy), the state value estimator (lr_td), the apical synaptic
# weights (lr_ap) and the pre-synaptic input to the apical dendrites (lr_trace).
HYPERPARAMETERS = {
    'biological': dict(theta_0=2.35, lr_policy=0.004, lr_td=0.00044, lr_ap=0.112, lr_trace=0.256),
    'fast': dict(theta_0=3., lr_policy=0.016, lr_td=0.0035, lr_ap=0.512, lr_trace=0.256),
}

# Learning rates swept one at a time around the values of the fast-learning regime (see get_learning_rates)
LEARNING_RATES = ('lr_td', 'lr_policy', 'lr_ap', 'lr_trace')

# Keys identifying various outcomes of the simulation
Key = Literal[
    'outcome', 'correct_trials', 'performance', 'learning_t', 'expert_t', 'v_hat', 'td_delta', 'w_ap', 'x_pre_t',
    'sensory_dendrite', 'sensory_dendrite_t1', 'sensory_dendrite_t2', 'outcome_dendrite', 'gain', 'w_bas',
    'w_bas_final', 'w_bas_history', 'w_bas_history_t', 'x_som_texture', 'x_som_cue'
]
K_OUTCOME: Key = 'outcome'
K_CORRECT_TRIALS: Key = 'correct_trials'
K_PERFORMANCE: Key = 'performance'
K_LEARNING_T: Key = 'learning_t'
K_EXPERT_T: Key = 'expert_t'
K_V_HAT: Key = 'v_hat'
K_TD_DELTA: Key = 'td_delta'
K_TIMINGS: Key = 'x_pre_t'
K_W_AP: Key = 'w_ap'
K_W_BAS: Key = 'w_bas'
K_W_BAS_FINAL: Key = 'w_bas_final'
K_W_BAS_H: Key = 'w_bas_history'
K_W_BAS_T: Key = 'w_bas_history_t'
K_X_SOM_TXT: Key = 'x_som_texture'
K_X_SOM_CUE: Key = 'x_som_cue'
K_DENDRITE_SENSORY: Key = 'sensory_dendrite'
K_DENDRITE_SENSORY_T1: Key = 'sensory_dendrite_t1'
K_DENDRITE_SENSORY_T2: Key = 'sensory_dendrite_t2'
K_DENDRITE_OUTCOME: Key = 'outcome_dendrite'
K_GAIN: Key = 'gain'
KEYS_1D = [K_OUTCOME, K_CORRECT_TRIALS, K_PERFORMANCE, K_DENDRITE_OUTCOME]
KEYS_2D = [K_V_HAT, K_TD_DELTA, K_TIMINGS, K_W_AP, K_DENDRITE_SENSORY, K_DENDRITE_SENSORY_T1, K_DENDRITE_SENSORY_T2,
           K_X_SOM_TXT, K_X_SOM_CUE, K_GAIN]


class Outcome(Enum):
    """
    Enumeration class for the four possible trial outcomes in the go/no-go sensory discrimination task
    """
    HIT = 0
    CR = 1
    FA = 2
    MISS = 3


class Simulation(Enum):
    """
    Enumeration class for the types of simulations implemented: the model without bottom-up plasticity, with active
    apical dendrites (DEFAULT) or with the apical dendrites inhibited for the first N_TRIALS trials (APICAL_INHIBITION),
    and the model with one of four bottom-up (basal) plasticity rules (BU_HEBB, BU_BCM, BU_CALCIUM, BU_APICAL)
    """
    DEFAULT = 0
    APICAL_INHIBITION = 1
    BU_HEBB = 2
    BU_BCM = 3
    BU_CALCIUM = 4
    BU_APICAL = 5


# Bottom-up plasticity simulations
BU_SIMULATIONS = (Simulation.BU_HEBB, Simulation.BU_BCM, Simulation.BU_CALCIUM, Simulation.BU_APICAL)

# Bottom-up plasticity rules in each learning regime: the bound that keeps the basal weights in [0, 1] ('soft': the
# update is scaled by (1 - w) w, 'pre': each stimulus keeps its initial total weight onto all neurons, 'l1': each neuron
# keeps its initial total basal input), the basal learning rate (lr_bas) and the constants of the rule (see
# simulations.basal_rule). BU_APICAL has no setting in the fast regime: at the TD learning rate of that regime the state
# value estimate diverges at every lr_bas
RULE_HYPERPARAMETERS = {
    'biological': {
        Simulation.BU_HEBB: dict(bound='soft', lr_bas=0.512, theta_bas_0=7.),
        Simulation.BU_BCM: dict(bound='soft', lr_bas=0.128, bcm_tau=200., bcm_e0_scale=0.25),
        Simulation.BU_CALCIUM: dict(bound='pre', lr_bas=0.016, ca_theta_p=1.5, ca_ltd_frac=0.4, ca_gamma_d=2.,
                                    ca_beta=2.),
        Simulation.BU_APICAL: dict(bound='l1', lr_bas=0.005, xbar_tau=50.),
    },
    'fast': {
        Simulation.BU_HEBB: dict(bound='soft', lr_bas=0.512, theta_bas_0=10.),
        Simulation.BU_BCM: dict(bound='soft', lr_bas=0.256, bcm_tau=50., bcm_e0_scale=0.25),
        Simulation.BU_CALCIUM: dict(bound='l1', lr_bas=0.001, ca_theta_p=2., ca_ltd_frac=0.5, ca_gamma_d=2.,
                                    ca_beta=2.),
    },
}


def apical_transfer(x) -> torch.Tensor:
    """
    Transfer function taking apical input and returning the apical activation
    :param x: apical input tensor
    :return: apical activation tensor
    """
    return torch.sigmoid((MAX_GAIN-1)*x-torch.log(torch.tensor(MAX_GAIN-1)))


def gain(x: torch.Tensor) -> torch.Tensor:
    """
    Transfer function taking apical activation and returning a multiplicative gain
    :param x: apical activation
    :return: multiplicative gain
    """
    return 1 + MAX_GAIN * torch.clip(x - 1. / MAX_GAIN, 0, (MAX_GAIN-1) / MAX_GAIN)


def time_to_time_step(t: float) -> int:
    """
    Convert time in seconds to time step in the simulation.
    :param t: Time in seconds
    :return: Time step in the simulation
    """
    return int(t * HZ - TONE_T)


def time_step_to_time(t: int) -> float:
    """
    Convert time step in the simulation to time in seconds.
    :param t: Time step in the simulation
    :return: Time in seconds
    """
    return (t + TONE_T) / HZ


def get_hyperparameters(simulation: Simulation = Simulation.DEFAULT, fast: bool = False) -> dict:
    """
    Get the hyperparameters of a simulation.
    :param simulation: Simulation identifier
    :param fast: Whether to use the hyperparameters of the fast-learning regime instead of the biological ones
    :return: Dictionary of the learning rates and theta_0 of the learning regime, plus the bound, basal learning rate
    and constants of the bottom-up plasticity rule for the bottom-up plasticity simulations
    """
    regime = 'fast' if fast else 'biological'
    if simulation in BU_SIMULATIONS and simulation not in RULE_HYPERPARAMETERS[regime]:
        raise ValueError(f'{simulation} has no setting in the {regime} regime')
    return {**HYPERPARAMETERS[regime], **RULE_HYPERPARAMETERS[regime].get(simulation, {})}


def get_bu_simulations(fast: bool = False) -> tuple[Simulation, ...]:
    """
    Get the bottom-up plasticity simulations of a learning regime.
    :param fast: Whether to get those of the fast-learning regime instead of the biological one
    :return: The bottom-up plasticity simulations with a setting in RULE_HYPERPARAMETERS for the regime
    """
    return tuple(s for s in BU_SIMULATIONS if s in RULE_HYPERPARAMETERS['fast' if fast else 'biological'])


def is_inhibited(simulation: Simulation, inhibition: bool = False) -> bool:
    """
    Whether the apical dendrites of a simulation are inhibited for the first N_TRIALS trials.
    :param simulation: Simulation identifier
    :param inhibition: Whether the apical dendrites are inhibited (only for the bottom-up plasticity simulations)
    :return: True for Simulation.APICAL_INHIBITION and for the bottom-up plasticity simulations with inhibition
    """
    if inhibition and simulation not in BU_SIMULATIONS:
        raise ValueError(f'Apical inhibition is an option of the bottom-up plasticity simulations only, use '
                         f'Simulation.APICAL_INHIBITION instead of {simulation} with inhibition')
    return simulation == Simulation.APICAL_INHIBITION or inhibition


def get_n_trials(simulation: Simulation = Simulation.DEFAULT, inhibition: bool = False) -> int:
    """
    Get the number of trials of a simulation.
    :param simulation: Simulation identifier
    :param inhibition: Whether the apical dendrites are inhibited (only for the bottom-up plasticity simulations)
    :return: Number of trials
    """
    return N_TRIALS_AP_INH if is_inhibited(simulation=simulation, inhibition=inhibition) else N_TRIALS


def get_path(simulation: Simulation = Simulation.DEFAULT, fast: bool = False, inhibition: bool = False,
             theta_0: float = None, lr: tuple[str, float] = None) -> Path:
    """
    Get the path for the pickle file containing the simulation outcomes.
    :param simulation: Simulation identifier.
    :param fast: Whether the simulation uses the hyperparameters of the fast-learning regime
    :param inhibition: Whether the apical dendrites are inhibited (only for the bottom-up plasticity simulations)
    :param theta_0: Value of the theta_0 parameter in the theta_0 sweep of the default simulation (None otherwise)
    :param lr: Name and value of the learning rate in a learning-rate sweep of the default simulation (None otherwise)
    :return: Path of the file containing the simulation outcomes
    """
    results_dir = ROOT_DIR / 'results' / ('fast' if fast else '')
    inhibited = is_inhibited(simulation=simulation, inhibition=inhibition)
    if (theta_0 is not None or lr is not None) and simulation != Simulation.DEFAULT:
        raise ValueError(f'The theta_0 and learning-rate sweeps only exist for {Simulation.DEFAULT}')
    if theta_0 is not None:
        path = results_dir / f'theta_0/t_{theta_0}.pickle'
    elif lr is not None:
        name, value = lr
        path = results_dir / f'{name}/{name}_{value:g}.pickle'
    elif simulation == Simulation.DEFAULT:
        path = results_dir / 'results_default.pickle'
    elif simulation == Simulation.APICAL_INHIBITION:
        path = results_dir / 'results_perturbed.pickle'
    elif simulation in BU_SIMULATIONS:
        path = results_dir / f"results_{get_rule_name(simulation)}{'_perturbed' if inhibited else ''}.pickle"
    else:
        raise ValueError(simulation)

    return path


def get_rule_name(simulation: Simulation) -> str:
    """
    Get the name of the bottom-up plasticity rule of a simulation.
    :param simulation: Simulation identifier
    :return: 'hebb', 'bcm', 'calcium' or 'apical'
    """
    return {Simulation.BU_HEBB: 'hebb', Simulation.BU_BCM: 'bcm', Simulation.BU_CALCIUM: 'calcium',
            Simulation.BU_APICAL: 'apical'}[simulation]


def get_runs() -> list[tuple[Simulation, bool, bool]]:
    """
    All the simulation runs apart from the theta_0 and learning-rate sweeps.
    :return: List of (simulation, fast, inhibition) tuples
    """
    runs = []
    for fast in (False, True):
        runs += [(Simulation.DEFAULT, fast, False), (Simulation.APICAL_INHIBITION, fast, False)]
        for simulation in get_bu_simulations(fast=fast):
            runs += [(simulation, fast, False), (simulation, fast, True)]
    return runs


def get_run_name(simulation: Simulation, fast: bool = False, inhibition: bool = False) -> str:
    """
    Get the name of a simulation run (for log messages).
    :param simulation: Simulation identifier
    :param fast: Whether the simulation uses the hyperparameters of the fast-learning regime
    :param inhibition: Whether the apical dendrites are inhibited (only for the bottom-up plasticity simulations)
    :return: Name of the run, e.g. 'default', 'apical_inhibition', 'hebb_apical_inhibition' or 'fast_bcm'
    """
    if simulation == Simulation.DEFAULT:
        name = 'default'
    elif simulation == Simulation.APICAL_INHIBITION:
        name = 'apical_inhibition'
    elif simulation in BU_SIMULATIONS:
        name = get_rule_name(simulation) + ('_apical_inhibition' if inhibition else '')
    else:
        raise ValueError(simulation)
    return ('fast_' if fast else '') + name


def get_thetas_0(fast: bool = False) -> list[float]:
    """
    The values of theta_0 that are tried in the theta_0 sweep of the default simulation: 1 to 5 in steps of 0.5 and the
    value of the learning regime (at the biological learning rates no seed learns within N_TRIALS trials from
    theta_0 = 5 on; at the fast ones the state value estimate diverges in every seed up to theta_0 = 2 and in 2 of 10
    seeds at 2.5).
    :param fast: Whether to get the values of the sweep in the fast-learning regime instead of the biological one
    :return: List of theta_0 values
    """
    grid = [1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.]
    return sorted(set(grid + [HYPERPARAMETERS['fast' if fast else 'biological']['theta_0']]))


def get_learning_rates(name: str) -> list[float]:
    """
    The values that are tried in the sweep of a learning rate of the default simulation in the fast-learning regime, the
    other hyperparameters at the values of the regime: the value of the regime times 2^(k/4) for k = -12, -10, ..., 12
    (half-octave steps from 1/8 to 8 times) and k = -1 and 1 (quarter-octave steps next to the value of the regime),
    except k = 0 (the default simulation itself); for lr_trace only the values below 1 (at 1 the presynaptic input to
    the apical dendrites is the last absolute TD error, above 1 its update overshoots) and 1 itself.
    :param name: Name of the learning rate (lr_td, lr_policy, lr_ap or lr_trace, see HYPERPARAMETERS)
    :return: List of the values of the learning rate, in increasing order
    """
    exponents = sorted([k / 4 for k in range(-12, 13, 2) if k != 0] + [-1 / 4, 1 / 4])
    values = [HYPERPARAMETERS['fast'][name] * 2 ** e for e in exponents]
    if name == 'lr_trace':
        values = [v for v in values if v < 1.] + [1.]
    return values


def nan_array(shape: tuple[int, ...], dtype: type = np.float64) -> np.ndarray:
    """
    Make a numpy array filled with nan.
    :param shape: Tuple of integers corresponding to the shape of the array
    :param dtype: Data type of the array
    :return: Numpy array of nan.
    """
    return np.full(shape=shape, fill_value=np.nan, dtype=dtype)


def makedirs() -> None:
    """
    Creates directories in which simulation outcomes and panels are saved.
    """
    for subdir in ['results/theta_0', 'results/fast/theta_0', 'panels/svg', 'panels/fast/svg'] + [
            f'results/fast/{name}' for name in LEARNING_RATES]:
        p = ROOT_DIR / subdir
        p.mkdir(parents=True, exist_ok=True)


# Estimator of the learning curve of Smith et al. (2004): Smith, A.C., Frank, L.M., Wirth, S., Yanike, M., Hu, D.,
# Kubota, Y., Graybiel, A.M., Suzuki, W.A. and Brown, E.N. Dynamic analysis of learning in behavioral experiments.
# Journal of Neuroscience, 24(2), 447-461. Python port of the authors' Matlab code (runanalysisv3.m with forwardfilter.m,
# backwardfilter.m, em_bino.m, newtonsolve.m and pdistnv2.m), line by line including the convergence criteria, except
# that the confidence limits are the exact quantiles of the logit-normal posterior instead of the quantiles of 10000
# samples. For the trial outcomes of the original manuscript simulations, the learning and expert trials were identical
# to the Matlab output and the performance traces differed by at most 3e-3, the Monte Carlo noise of the Matlab version.

def _newton_solve(mu: float, x_old: float, sig_old_sq: float, n: float, n_max: int) -> tuple[float, int]:
    """
    Posterior mode of the learning state for one trial (newtonsolve.m).
    :param mu: Logit of the probability of a correct response by chance
    :param x_old: One-step prediction of the learning state
    :param sig_old_sq: Variance of the one-step prediction
    :param n: Number of correct responses in the trial
    :param n_max: Maximal number of correct responses in a trial
    :return: Posterior mode of the learning state and a flag that is 1 if the Newton iterations did not converge
    """
    em = np.exp(mu)
    for x0 in (None, -1.0, 1.0):
        it = x_old + sig_old_sq * (n - n_max * em * np.exp(x_old) / (1 + em * np.exp(x_old))) if x0 is None else x0
        for _ in range(100):
            e = em * np.exp(it)
            g = x_old + sig_old_sq * (n - n_max * e / (1 + e)) - it
            gp = -n_max * sig_old_sq * e / (1 + e) ** 2 - 1
            new = it - g / gp
            if abs(new - it) < 1e-10:
                return new, 0
            it = new
    return it, 1


def _forward_filter(responses: np.ndarray, n_max: int, sig_e: float, x_guess: float, sig_sq_guess: float, mu: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward recursive filtering of the learning state (forwardfilter.m).
    :param responses: Number of correct responses in each trial
    :param n_max: Maximal number of correct responses in a trial
    :param sig_e: Standard deviation of the learning state process
    :param x_guess: Initial learning state
    :param sig_sq_guess: Variance of the initial learning state
    :param mu: Logit of the probability of a correct response by chance
    :return: Filtered learning states x{k|k} and their variances, one-step predictions x{k|k-1} and their variances,
    each with one entry more than the number of trials
    """
    n_trials = len(responses)
    x_hat, sig_sq = np.zeros(n_trials + 1), np.zeros(n_trials + 1)
    x_old, sig_sq_old = np.zeros(n_trials + 1), np.zeros(n_trials + 1)
    x_hat[0], sig_sq[0] = x_guess, sig_sq_guess
    em = np.exp(mu)
    for k in range(1, n_trials + 1):
        x_old[k] = x_hat[k - 1]
        sig_sq_old[k] = sig_sq[k - 1] + sig_e ** 2
        x_hat[k], _ = _newton_solve(mu, x_old[k], sig_sq_old[k], responses[k - 1], n_max)
        e = em * np.exp(x_hat[k])
        denominator = -1 / sig_sq_old[k] - n_max * e / (1 + e) ** 2
        sig_sq[k] = -1 / denominator
    return x_hat, sig_sq, x_old, sig_sq_old


def _backward_filter(x: np.ndarray, x_old: np.ndarray, sig_sq: np.ndarray, sig_sq_old: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fixed-interval smoothing of the learning state (backwardfilter.m).
    :param x: Filtered learning states
    :param x_old: One-step predictions of the learning state
    :param sig_sq: Variances of the filtered learning states
    :param sig_sq_old: Variances of the one-step predictions
    :return: Smoothed learning states x{k|K}, their variances and the smoother gains A{k} (A[0] is unused, as A(1) in
    Matlab)
    """
    n = len(x)
    x_new, sig_new_sq, a = np.zeros(n), np.zeros(n), np.zeros(n)
    x_new[n - 1], sig_new_sq[n - 1] = x[n - 1], sig_sq[n - 1]
    for i in range(n - 2, 0, -1):
        a[i] = sig_sq[i] / sig_sq_old[i + 1]
        x_new[i] = x[i] + a[i] * (x_new[i + 1] - x_old[i + 1])
        sig_new_sq[i] = sig_sq[i] + a[i] ** 2 * (sig_new_sq[i + 1] - sig_sq_old[i + 1])
    return x_new, sig_new_sq, a


def _em_bino(x_new: np.ndarray, sig_new_sq: np.ndarray, a: np.ndarray, start_flag: int) -> float:
    """
    Maximization step for the variance of the learning state process (em_bino.m).
    :param x_new: Smoothed learning states
    :param sig_new_sq: Variances of the smoothed learning states
    :param a: Smoother gains
    :param start_flag: Treatment of the initial condition (updater_flag of smith_analysis)
    :return: New estimate of the variance of the learning state process
    """
    m = len(x_new)
    x_new_t, x_new_t_m1, sig_new_sq_t = x_new[2:m], x_new[1:m - 1], sig_new_sq[2:m]
    cov_calc = sig_new_sq_t * a[1:m - 1]  # Matlab A(k) is a[k - 1] here; A(2:end) pairs signewsq(k) with A(k - 1)
    term1 = np.sum(x_new_t ** 2) + np.sum(sig_new_sq_t)
    term2 = np.sum(cov_calc) + np.sum(x_new_t * x_new_t_m1)
    if start_flag == 1:
        term3, term4 = 1.5 * x_new[1] ** 2 + 2.0 * sig_new_sq[1], x_new[-1] ** 2 + sig_new_sq[-1]
    elif start_flag == 0:
        term3, term4 = 2 * x_new[1] ** 2 + 2 * sig_new_sq[1], x_new[-1] ** 2 + sig_new_sq[-1]
    else:
        term3, term4 = x_new[1] ** 2 + 2 * sig_new_sq[1], x_new[-1] ** 2 + sig_new_sq[-1]
        m = m - 1
    return (2 * (term1 - term2) + term3 - term4) / m


def smith_analysis(responses: np.ndarray, max_response: int = 1, background_prob: float = 0.5, sig_e: float = 0.005,
                   updater_flag: int = 2, max_steps: int = 4000) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Estimate the learning curve of one agent and its learning and expert trials with the method of Smith et al. (2004)
    (runanalysisv3.m).
    :param responses: Number of correct responses in each trial (0 or 1 for binary outcomes)
    :param max_response: Maximal number of correct responses in a trial
    :param background_prob: Probability of a correct response by chance
    :param sig_e: Initial standard deviation of the learning state process
    :param updater_flag: Treatment of the initial condition (0: fixed, 1: estimated, 2: removed from the likelihood)
    :param max_steps: Maximal number of iterations of the expectation maximization
    :return: Tuple of the learning trial (50 plus the first trial after trial 51 at which the lower 5 percent confidence
    bound of the probability of a correct response is above chance), the expert trial (last trial at which that bound is
    below chance), both counted from 1 and nan if not reached, the median performance trace and its lower 5 percent
    confidence bound (each with one entry more than the number of trials, the first being the prior)
    """
    responses = np.asarray(responses, dtype=float).ravel()
    mu = np.log(background_prob / (1 - background_prob))
    convergence = 1e-8
    x_guess, sig_sq_guess = 0.0, sig_e ** 2
    new_sig_sq, x_new_1_save = [], []
    x_new = sig_new_sq = None
    for i in range(max_steps):
        x, s, x_old, s_old = _forward_filter(responses, max_response, sig_e, x_guess, sig_sq_guess, mu)
        x_new, sig_new_sq, a = _backward_filter(x, x_old, s, s_old)
        if updater_flag == 1:
            x_new[0], sig_new_sq[0] = 0.5 * x_new[1], sig_e ** 2
        elif updater_flag == 0:
            x_new[0], sig_new_sq[0] = 0.0, sig_e ** 2
        else:
            x_new[0], sig_new_sq[0] = x_new[1], sig_new_sq[1]
        new_sig_sq.append(_em_bino(x_new, sig_new_sq, a, updater_flag))
        x_new_1_save.append(x_new[0])
        if i > 0:
            a1, a2 = abs(new_sig_sq[i] - new_sig_sq[i - 1]), abs(x_new_1_save[i] - x_new_1_save[i - 1])
            if (a1 < convergence and a2 < convergence and updater_flag >= 1) or \
                    (a1 < convergence and updater_flag == 0):
                break
        sig_e, x_guess, sig_sq_guess = np.sqrt(new_sig_sq[i]), x_new[0], sig_new_sq[0]

    # Confidence limits of the probability of a correct response (pdistnv2.m), with the exact logit-normal quantiles
    sd = np.sqrt(sig_new_sq)
    p05 = special.expit(mu + x_new + sd * stats.norm.ppf(0.05))
    p_mid = special.expit(mu + x_new)
    n = len(responses)
    below = np.where(p05 < background_prob)[0] + 1  # Trials counted from 1 as in Matlab
    t_expert = float(below[-1]) if len(below) and below[-1] < n + 1 else np.nan
    above = np.where(p05[51:] > background_prob)[0] + 1
    t_learn = float(50 + above[0]) if len(above) and above[0] < n + 1 else np.nan
    return t_learn, t_expert, p_mid, p05
