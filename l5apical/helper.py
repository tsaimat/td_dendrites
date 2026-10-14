"""
Helper functions and constants for the L5 apical dendrite model of sensory learning
"""

import torch
import numpy as np
from typing import Literal
from enum import Enum
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()  # Root directory
SMITH_DIR = ROOT_DIR / 'Smith'  # Directory containing matlab scripts and matlab derived performance traces

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
T1_IDX = 2  # Index of the texture 1 stimulus
T2_IDX = 1  # Index of the texture 2 stimulus
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

# Keys identifying various outcomes of the simulation
Key = Literal[
    'outcome', 'correct_trials', 'performance', 'learning_t', 'expert_t', 'v_hat', 'td_delta', 'w_ap', 'x_pre_t',
    'sensory_dendrite', 'sensory_dendrite_t1', 'sensory_dendrite_t2', 'outcome_dendrite', 'gain', 'w_bas',
    'w_bas_final', 'w_bas_history', 'w_bas_history_t', 'x_som_texture'
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
K_DENDRITE_SENSORY: Key = 'sensory_dendrite'
K_DENDRITE_SENSORY_T1: Key = 'sensory_dendrite_t1'
K_DENDRITE_SENSORY_T2: Key = 'sensory_dendrite_t2'
K_DENDRITE_OUTCOME: Key = 'outcome_dendrite'
K_GAIN: Key = 'gain'
KEYS_1D = [K_OUTCOME, K_CORRECT_TRIALS, K_PERFORMANCE, K_DENDRITE_OUTCOME]
KEYS_2D = [K_V_HAT, K_TD_DELTA, K_TIMINGS, K_W_AP, K_DENDRITE_SENSORY, K_DENDRITE_SENSORY_T1, K_DENDRITE_SENSORY_T2,
           K_X_SOM_TXT, K_GAIN]


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
    the model with one of three bottom-up (basal) plasticity rules (BU_HEBB, BU_BCM, BU_CALCIUM), and the file of the
    performance traces extracted from all simulations (SMITH_PERFORMANCE)
    """
    DEFAULT = 0
    APICAL_INHIBITION = 1
    SMITH_PERFORMANCE = 2
    BU_HEBB = 3
    BU_BCM = 4
    BU_CALCIUM = 5


BU_SIMULATIONS = (Simulation.BU_HEBB, Simulation.BU_BCM, Simulation.BU_CALCIUM)  # Bottom-up plasticity simulations

# Bottom-up plasticity rules in each learning regime: the bound that keeps the basal weights in [0, 1] ('soft': the
# update is scaled by (1 - w) w, 'pre': each stimulus keeps its initial total weight onto all neurons, 'l1': each neuron
# keeps its initial total basal input), the basal learning rate (lr_bas) and the constants of the rule (see
# simulations.basal_rule)
RULE_HYPERPARAMETERS = {
    'biological': {
        Simulation.BU_HEBB: dict(bound='soft', lr_bas=0.512, theta_bas_0=7.),
        Simulation.BU_BCM: dict(bound='soft', lr_bas=0.128, bcm_tau=200., bcm_e0_scale=0.25),
        Simulation.BU_CALCIUM: dict(bound='pre', lr_bas=0.016, ca_theta_p=1.5, ca_ltd_frac=0.4, ca_gamma_d=2.,
                                    ca_beta=2.),
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
    return {**HYPERPARAMETERS[regime], **RULE_HYPERPARAMETERS[regime].get(simulation, {})}


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
             theta_0: float = None) -> Path:
    """
    Get the path for the pickle file containing the simulation outcomes.
    :param simulation: Simulation identifier.
    :param fast: Whether the simulation uses the hyperparameters of the fast-learning regime
    :param inhibition: Whether the apical dendrites are inhibited (only for the bottom-up plasticity simulations)
    :param theta_0: Value of the theta_0 parameter in the theta_0 sweep of the default simulation (None otherwise)
    :return: Path of the file containing the simulation outcomes
    """
    results_dir = ROOT_DIR / 'results' / ('fast' if fast else '')
    inhibited = is_inhibited(simulation=simulation, inhibition=inhibition)
    if theta_0 is not None:
        if simulation != Simulation.DEFAULT or fast:
            raise ValueError(f'The theta_0 sweep only exists for the biological {Simulation.DEFAULT}')
        path = results_dir / f'theta_0/t_{theta_0}.pickle'
    elif simulation == Simulation.DEFAULT:
        path = results_dir / 'results_default.pickle'
    elif simulation == Simulation.APICAL_INHIBITION:
        path = results_dir / 'results_perturbed.pickle'
    elif simulation == Simulation.SMITH_PERFORMANCE:
        path = SMITH_DIR / 'performances.mat'
    elif simulation in BU_SIMULATIONS:
        path = results_dir / f"results_{get_rule_name(simulation)}{'_perturbed' if inhibited else ''}.pickle"
    else:
        raise ValueError(simulation)

    return path


def get_rule_name(simulation: Simulation) -> str:
    """
    Get the name of the bottom-up plasticity rule of a simulation.
    :param simulation: Simulation identifier
    :return: 'hebb', 'bcm' or 'calcium'
    """
    return {Simulation.BU_HEBB: 'hebb', Simulation.BU_BCM: 'bcm', Simulation.BU_CALCIUM: 'calcium'}[simulation]


def get_runs() -> list[tuple[Simulation, bool, bool]]:
    """
    All the simulation runs whose performance traces are extracted, in the order of the files in Smith/getperfs.m and
    of the rows of expert_t and learning_t in Smith/performances.mat.
    :return: List of (simulation, fast, inhibition) tuples
    """
    runs = []
    for fast in (False, True):
        runs += [(Simulation.DEFAULT, fast, False), (Simulation.APICAL_INHIBITION, fast, False)]
        for simulation in BU_SIMULATIONS:
            runs += [(simulation, fast, False), (simulation, fast, True)]
    return runs


def get_run_name(simulation: Simulation, fast: bool = False, inhibition: bool = False) -> str:
    """
    Get the name of a simulation run in the Matlab files (Smith/outcomes_<name>.mat and perf_<name> in
    Smith/performances.mat).
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


def get_thetas_0() -> list[float]:
    """
    The values of theta_0 values that are tried in the theta_0 sweep of the default simulation (at the biological
    learning rates no seed learns within N_TRIALS trials from theta_0 = 5 on).
    :return: List of theta_0 values
    """
    return [1., 1.5, 2., 2.35, 2.5, 3., 3.5, 4., 4.5, 5.]


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
    for subdir in ['results/theta_0', 'results/fast', 'panels/svg']:
        p = ROOT_DIR / subdir
        p.mkdir(parents=True, exist_ok=True)
