"""
Functions to generate the figures of the paper from the simulation results in the 'results' directory
"""

import warnings
import matplotlib
import matplotlib.patheffects
import matplotlib.pyplot as plt
from math import floor
from pickle import load
from scipy import stats
from l5apical.parula import PARULA
from l5apical.helper import *

# Colors
COL_SENSORY = "#2985C4"
COL_ITI = '#cdcdcdff'
COL_PRE = '#004977ff'
COL_TEXTURE = "#ce583aff"
COL_T1 = 'k'
COL_T2 = '#9f9f9fff'
COL_DISTRACTOR = "#4cf3edff"
COL_OUTCOME = "#eda922ff"
COL_RESTORED = "#77c044ff"
COL_HIT = "#25b592ff"
COL_CR = "#794783f8"
GO_NOGO_MAP = matplotlib.colors.LinearSegmentedColormap.from_list(name="", colors=[COL_T2, COL_T1])
COL_HEBB = '#2a78d6'  # Colors of the bottom-up plasticity rules (distinguishable with color vision deficiencies)
COL_BCM = '#eb6834'
COL_CALCIUM = '#4a3aa7'
COL_APICAL = '#1baf7a'
COL_FAST = '#2a78d6'  # Color of the fast-learning regime (against grey for the biological one)
COL_DIVERGED = '0.9'  # Shading of the parameter values at which the simulation diverges in every seed
COL_PARTLY_DIVERGED = '0.955'  # Shading of those at which it diverges in some seeds but not all

# Dimensions for panels
INCH = 2.54
DPI = 450
FIG_W_PAD = 0.1  # inches
FIG_H_PAD = 0.06  # inches
PANEL_LABEL_HEIGHT = 0.11110236
AXIS_WIDTH = 0.5
LINE_WIDTH = 0.7
MARKER_SIZE = 2
FIG_WIDTH = 16 / INCH
PANEL_WIDTH = FIG_WIDTH / 3.
PANEL_HEIGHT = 4.5 / INCH
FONT_SIZE = 6

# Plot directories (the panels of the fast-learning regime in a subdirectory)
PLOT_DIR = ROOT_DIR / "panels"
FAST_DIR = 'fast'
SVG_DIR = 'svg'

# Processing initializations
N_BEFORE_EXPERT = 500  # Number of trials before the expert trial to plot
N_EXPERTS = 150  # Number of expert trials to plot
N_AV_BIN = 20  # Number of available bins for histograms
BIN_MAX = 6  # Maximal bin ID for histogram
RESPONSIVE_THRESHOLD = 0.1  # Mean somatic activity in the texture window above which a neuron is texture-responsive
SELECTIVE_THRESHOLD = 0.2  # Difference of that activity between go and no-go trials beyond which it is selective
N_PHASE_TRIALS = 100  # Number of trials over which the responses before and after learning are averaged
END_T = OUTCOME_T + 2  # Time step up to which to plot
DETECTOR_THRESHOLD = 0.5  # Basal weight above which an input makes a neuron a detector of that input (half the maximum)
RECRUITED_THRESHOLD = 0.5  # Final apical weight above which a neuron counts as recruited by the apical loop

# Labels and colors of the control without bottom-up plasticity (Simulation.DEFAULT, Simulation.APICAL_INHIBITION in the
# apical inhibition protocol) and of the bottom-up plasticity simulations
BU_LABELS = {Simulation.DEFAULT: 'Control', Simulation.BU_HEBB: 'Hebb', Simulation.BU_BCM: 'BCM',
             Simulation.BU_CALCIUM: 'Calcium', Simulation.BU_APICAL: 'Plastic $w^\\mathrm{bas}$'}
BU_COLORS = {Simulation.DEFAULT: 'grey', Simulation.BU_HEBB: COL_HEBB, Simulation.BU_BCM: COL_BCM,
             Simulation.BU_CALCIUM: COL_CALCIUM, Simulation.BU_APICAL: COL_APICAL}

# Labels and colors of the learning regimes (keyed by the fast option)
REGIME_LABELS = {False: 'Biological', True: 'Fast'}
REGIME_COLORS = {False: 'grey', True: COL_FAST}
# Names of the learning rates of the fast-learning regime in their sweeps (fig. S15e)
LR_LABELS = {'lr_td': 'TD', 'lr_policy': 'Policy', 'lr_ap': 'Apical weights', 'lr_trace': 'Apical input'}


def get_plot_dir(fast: bool = False) -> Path:
    """
    Get the directory in which the panels of a learning regime are saved
    :param fast: whether to get the directory of the fast-learning regime (the subdirectory FAST_DIR of PLOT_DIR)
    :return: directory of the panels
    """
    return PLOT_DIR / FAST_DIR if fast else PLOT_DIR


def get_results(simulation: Simulation = Simulation.DEFAULT, expert_aligned: bool = False, theta_0: float = None,
                fast: bool = False, inhibition: bool = False, lr: tuple[str, float] = None) -> list[dict]:
    """
    Align simulation outcomes to the trials at which the simulated agents crossed the expertise thresholds
    :param simulation: Enum identifier of the simulation type
    :param expert_aligned: Whether to align trials to the first expert trial
    :param theta_0: Value of the theta_0 parameter (only for the theta_0 sweep of the default simulation)
    :param fast: Whether to load the simulation with the hyperparameters of the fast-learning regime
    :param inhibition: Whether to load the simulation with apical inhibition (only for bottom-up plasticity simulations)
    :param lr: Name and value of the learning rate (only for the learning-rate sweeps of the default simulation)
    :return: results dictionary with values aligned to expert times
    """
    with open(get_path(simulation=simulation, fast=fast, inhibition=inhibition, theta_0=theta_0, lr=lr),
              'rb') as handle:
        results = load(handle)
    if expert_aligned:
        for s in range(N_SEEDS):
            expert_t = int(results[s][K_EXPERT_T])
            for k in [K_LEARNING_T, K_EXPERT_T]:
                results[s][k] = results[s][k] - expert_t + N_BEFORE_EXPERT
            for k in [k for k in KEYS_1D + KEYS_2D if k in results[s]]:
                results[s][k] = align_to_expert(values=results[s][k], expert_t=expert_t)

    return results


def get_bu_results(simulation: Simulation, inhibition: bool = False, fast: bool = False) -> list[dict]:
    """
    Load the results of the control without bottom-up plasticity or of a bottom-up plasticity simulation, in the
    protocol with or without apical inhibition
    :param simulation: Simulation.DEFAULT for the control or one of the bottom-up plasticity simulations
    :param inhibition: Whether to load the protocol in which the apical dendrites are inhibited for the first N_TRIALS
    trials (Simulation.APICAL_INHIBITION for the control)
    :param fast: Whether to load the simulation with the hyperparameters of the fast-learning regime
    :return: results dictionaries of all seeds (not aligned)
    """
    if simulation == Simulation.DEFAULT:
        return get_results(simulation=Simulation.APICAL_INHIBITION if inhibition else Simulation.DEFAULT, fast=fast)
    return get_results(simulation=simulation, fast=fast, inhibition=inhibition)


def get_bu_conditions(fast: bool = False) -> tuple[Simulation, ...]:
    """
    Get the conditions compared in the figures of the bottom-up plasticity rules: the control without bottom-up
    plasticity (Simulation.DEFAULT) and the bottom-up plasticity simulations of the learning regime
    :param fast: Whether to get the conditions of the fast-learning regime instead of the biological one
    :return: Simulation identifiers of the conditions, the control first
    """
    return (Simulation.DEFAULT,) + get_bu_simulations(fast=fast)


def align_to_expert(values: np.ndarray, expert_t: int, n_before: int = N_BEFORE_EXPERT) -> np.ndarray:
    """
    Get the values of the n_before trials before the expert trial and of the N_EXPERTS trials from it on, nan for the
    trials outside the simulation
    :param values: array of values with the trials along the first dimension
    :param expert_t: expert trial
    :param n_before: number of trials before the expert trial
    :return: array of the n_before + N_EXPERTS trials aligned to the expert trial
    """
    first = expert_t - n_before
    aligned = np.full((n_before + N_EXPERTS,) + values.shape[1:], np.nan)
    start, stop = max(first, 0), min(expert_t + N_EXPERTS, len(values))
    aligned[start - first:stop - first] = values[start:stop]
    return aligned


def get_trials(values: list[np.ndarray], results: list[dict], expert_aligned: bool) -> list[np.ndarray]:
    """
    Get the values of each seed for all its trials or for the trials aligned to its expert trial
    :param values: arrays of values of each seed with the trials along the first dimension (not aligned)
    :param results: results dictionaries of all seeds (not aligned)
    :param expert_aligned: whether to align the trials to the expert trial of each seed
    :return: list of the arrays of values of each seed
    """
    if expert_aligned:
        return [align_to_expert(values=v, expert_t=int(r[K_EXPERT_T])) for v, r in zip(values, results)]
    return values


def get_diverged(result: dict) -> bool:
    """
    Whether the simulation of a seed diverged (see simulations.simulate_seed), which leaves its last trials without
    outcome
    :param result: results dictionary of one seed (not aligned)
    :return: True if the simulation stopped before its last trial
    """
    return bool(np.isnan(result[K_OUTCOME][-1]))


def get_texture_responsive(result: dict, trials: tuple[int, int] = (0, N_PHASE_TRIALS)) -> np.ndarray:
    """
    Get the texture-responsive neurons, whose mean somatic activity in the texture window over a range of trials is
    above RESPONSIVE_THRESHOLD
    :param result: results dictionary of one seed (not aligned)
    :param trials: first trial and last trial plus one of the range
    :return: boolean array encoding whether each neuron is texture-responsive
    """
    return np.array([np.mean(result[K_X_SOM_TXT][trials[0]:trials[1], i]) > RESPONSIVE_THRESHOLD for i in range(N_Z)])


def get_texture_selectivity(result: dict, trials: tuple[int, int] = (0, N_PHASE_TRIALS)) -> np.ndarray:
    """
    Get the texture selectivity of each neuron: its mean somatic activity in the texture window over a range of trials
    in go trials minus that in no-go trials
    :param result: results dictionary of one seed (not aligned)
    :param trials: first trial and last trial plus one of the range
    :return: array of the selectivity of each neuron
    """
    values = result[K_X_SOM_TXT][trials[0]:trials[1]].astype(np.float64)
    outcomes = result[K_OUTCOME][trials[0]:trials[1]]
    return (np.nanmean(get_texture_specific(texture=True, outcomes=outcomes, values=values), axis=0)
            - np.nanmean(get_texture_specific(texture=False, outcomes=outcomes, values=values), axis=0))


def get_texture_selective(result: dict, trials: tuple[int, int] = (0, N_PHASE_TRIALS)
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the go-selective and the no-go-selective neurons: the texture-responsive neurons whose mean somatic activity in
    the texture window over a range of trials is larger in go than in no-go trials by more than SELECTIVE_THRESHOLD, or
    smaller by more than it (the selective bins of fig. S12d)
    :param result: results dictionary of one seed (not aligned)
    :param trials: first trial and last trial plus one of the range
    :return: boolean arrays encoding whether each neuron is go-selective and whether it is no-go-selective
    """
    selectivity = get_texture_selectivity(result=result, trials=trials)
    responsive = get_texture_responsive(result=result, trials=trials)
    return responsive & (selectivity > SELECTIVE_THRESHOLD), responsive & (selectivity < -SELECTIVE_THRESHOLD)


def get_selectivity_classes(result: dict, trials: tuple[int, int] = (0, N_PHASE_TRIALS)) -> np.ndarray:
    """
    Get the fractions of go-selective, no-go-selective, non-selective (texture-responsive but neither go- nor
    no-go-selective) and unresponsive neurons over a range of trials (the classes of fig. S12d)
    :param result: results dictionary of one seed (not aligned)
    :param trials: first trial and last trial plus one of the range
    :return: array of the four fractions
    """
    responsive = get_texture_responsive(result=result, trials=trials)
    go, nogo = get_texture_selective(result=result, trials=trials)
    return np.array([np.mean(go), np.mean(nogo), np.mean(responsive & ~go & ~nogo), np.mean(~responsive)])


def get_cue_selective(result: dict, trials: tuple[int, int] = (0, N_PHASE_TRIALS)) -> np.ndarray:
    """
    Get the cue-selective neurons, whose mean somatic activity in the cue window over a range of trials is above
    RESPONSIVE_THRESHOLD and larger than in the texture window by more than SELECTIVE_THRESHOLD (the criterion of the
    go- and no-go-selective neurons, with the cue against the textures)
    :param result: results dictionary of one seed (not aligned)
    :param trials: first trial and last trial plus one of the range
    :return: boolean array encoding whether each neuron is cue-selective
    """
    cue = np.mean(result[K_X_SOM_CUE][trials[0]:trials[1]].astype(np.float64), axis=0)
    texture = np.mean(result[K_X_SOM_TXT][trials[0]:trials[1]].astype(np.float64), axis=0)
    return (cue > RESPONSIVE_THRESHOLD) & (cue - texture > SELECTIVE_THRESHOLD)


def get_sensory_dendrites(result: dict, neurons: np.ndarray) -> np.ndarray:
    """
    Get the activity of the sensory dendrites at each time step of each trial, averaged over a group of neurons and
    computed from the recorded sensory dendrite synaptic weights and pre-synaptic inputs the way the simulation records
    it for the go and no-go neurons (before the outcome with the values at the start of the trial)
    :param result: results dictionary of one seed (not aligned)
    :param neurons: boolean array or indices of the neurons
    :return: array (trials x time steps) of the mean sensory dendrite activity
    """
    outcome_t = OUTCOME_T - TONE_T
    w_ap = torch.tensor(result[K_W_AP][:, neurons], dtype=torch.float64)
    x_pre = torch.tensor(result[K_TIMINGS])
    activity = nan_array((len(x_pre), N_TIME_STEPS))
    activity[0, :outcome_t] = apical_transfer(torch.tensor(0.)).item()  # The pre-synaptic inputs start at zero
    activity[1:, :outcome_t] = torch.mean(apical_transfer(x_pre[:-1, :outcome_t, None] * w_ap[:-1, None, :]),
                                          dim=2).numpy()
    activity[:, outcome_t] = torch.mean(apical_transfer(x_pre[:, outcome_t, None] * w_ap), dim=1).numpy()
    activity[result[K_DENDRITE_OUTCOME] == 0, outcome_t] = 0.  # Trials with the outcome-time apical activity silenced
    return activity


def get_gains(result: dict, neurons: np.ndarray, time_step: int) -> np.ndarray:
    """
    Get the apical gains of a group of neurons at a time step before the outcome in each trial, computed from the
    recorded sensory dendrite synaptic weights and pre-synaptic inputs at the start of the trial (as in
    get_sensory_dendrites)
    :param result: results dictionary of one seed (not aligned)
    :param neurons: boolean array or indices of the neurons
    :param time_step: simulated time step (0 for the cue, TEXTURE_T - TONE_T for the texture)
    :return: array (trials x neurons) of the apical gains
    """
    w_ap = torch.tensor(result[K_W_AP][:, neurons], dtype=torch.float64)
    x_pre = torch.tensor(result[K_TIMINGS][:, time_step])
    gains = nan_array(tuple(w_ap.shape))
    gains[0] = gain(apical_transfer(torch.tensor(0.))).item()  # The pre-synaptic inputs start at zero
    gains[1:] = gain(apical_transfer(x_pre[:-1, None] * w_ap[:-1])).numpy()
    return gains


def get_basal_history(result: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the snapshots of the basal weights of one seed (after trials 0, SNAPSHOT_EVERY, ..., last), or the initial
    weights at the first and last trial for the simulations without bottom-up plasticity
    :param result: results dictionary of one seed (not aligned)
    :return: trials of the snapshots and the basal weights (snapshots x basal inputs x neurons)
    """
    if K_W_BAS_H in result:
        return result[K_W_BAS_T], result[K_W_BAS_H]
    return np.array([0, len(result[K_OUTCOME]) - 1]), np.stack([result[K_W_BAS]] * 2)


def get_texture_detectors(w_bas: np.ndarray) -> np.ndarray:
    """
    Get the texture detectors, the neurons with a basal weight above DETECTOR_THRESHOLD from either texture, whatever
    their other weights (the bottom-up plasticity rules drive many weights to the bound 1, so that several inputs of a
    neuron often share its largest weight)
    :param w_bas: basal weights (basal inputs x neurons)
    :return: boolean array encoding whether each neuron is a texture detector
    """
    return np.max(w_bas[[T1_IDX, T2_IDX]], axis=0) > DETECTOR_THRESHOLD


def get_recruited(result: dict) -> np.ndarray:
    """
    Get the neurons recruited by the apical loop, whose apical synaptic weight w_ap at the end of the simulation is
    above RECRUITED_THRESHOLD (the final weights are close to 0 or to 1, so that the threshold hardly matters)
    :param result: results dictionary of one seed (not aligned)
    :return: boolean array encoding whether each neuron is recruited
    """
    return result[K_W_AP][-1] > RECRUITED_THRESHOLD


def get_basal_change(result: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the change of the basal weights of each neuron from their initial values at each snapshot (see
    get_basal_history): the sum over the basal inputs of the neuron of the absolute differences to the initial weights
    :param result: results dictionary of one seed (not aligned)
    :return: trials of the snapshots and the changes (snapshots x neurons)
    """
    trials, w_bas = get_basal_history(result=result)
    return trials, np.sum(np.abs(w_bas.astype(np.float64) - result[K_W_BAS].astype(np.float64)), axis=1)


def get_input_class_weights(w_bas: np.ndarray, w_bas_0: np.ndarray, neurons: np.ndarray) -> np.ndarray:
    """
    Get the mean basal weights of a group of neurons from four classes of inputs: the tone, the preferred texture of
    each neuron (the texture with the larger initial weight; the initial weights of the two textures are never tied),
    its other texture and its distractors (the sum of their weights)
    :param w_bas: basal weights (snapshots x basal inputs x neurons)
    :param w_bas_0: initial basal weights (basal inputs x neurons), which define the preferred texture of each neuron
    :param neurons: boolean array encoding whether each neuron belongs to the group
    :return: array (snapshots x the four classes) of the weights averaged over the neurons of the group
    """
    preferred = np.where(w_bas_0[T1_IDX] >= w_bas_0[T2_IDX], T1_IDX, T2_IDX)[neurons]
    other = T1_IDX + T2_IDX - preferred
    w = w_bas[:, :, neurons].astype(np.float64)
    columns = np.arange(w.shape[2])
    return np.stack([w[:, TONE_IDX].mean(1), w[:, preferred, columns].mean(1), w[:, other, columns].mean(1),
                     w[:, N_TASK_STIMULI:].sum(1).mean(1)], axis=1)


def get_outcome_specific(outcome: Outcome, outcomes: type[np.ndarray | list[np.ndarray]],
                         values: type[np.ndarray | list[np.ndarray]]) -> type[np.ndarray | list[np.ndarray]]:
    """
    From the array of values for each trial, only keep the ones that occurred in trials of a specific outcome and
    replace the other values with nan
    :param outcome: specific outcome to select for
    :param outcomes: array or list of arrays containing the outcomes of each trial
    :param values: array or list of arrays containing the values to pick out depending on the trial outcome
    :return: array or list of arrays with only the value that occurred for the selected outcome
    """
    vs = np.copy(values)
    if isinstance(outcomes, list) and isinstance(values, list):
        if vs[0].ndim == 1:
            for s in range(len(outcomes)):
                vs[s][outcomes[s] != outcome.value] = np.nan
        elif vs[0].ndim == 2:
            for s in range(len(outcomes)):
                vs[s][outcomes[s] != outcome.value, :] = np.nan
        else:
            raise NotImplementedError(vs[0].ndim)

    elif isinstance(outcomes, np.ndarray) and isinstance(values, np.ndarray):
        if vs.ndim == 1:
            vs[outcomes != outcome.value] = np.nan
        elif vs.ndim == 2:
            vs[outcomes != outcome.value, :] = np.nan
        else:
            raise NotImplementedError(vs.ndim)
    else:
        raise NotImplementedError(f"Object types are {type(outcomes)} and {type(values)}")

    return vs


def get_texture_specific(texture: bool, outcomes: type[np.ndarray | list[np.ndarray]],
                         values: type[np.ndarray | list[np.ndarray]]) -> type[np.ndarray | list[np.ndarray]]:
    """
    From the array of values for each trial, only keep the ones that occurred in trials in which a specific texture was
    shown replace the other values with nan
    :param texture: boolean of the selected stimulus. True for go-stimulus (s1 / texture 1), False for nogo-stimulus
    :param outcomes: array or list of arrays containing the outcomes of each trial
    :param values: array or list of arrays containing the values to pick out depending on the trial outcome
    :return: array or list of arrays with only the value that occurred for the selected stimulus
    """
    vs = np.copy(values)
    if texture:
        bad_outcomes = [Outcome.FA.value, Outcome.CR.value]
    else:
        bad_outcomes = [Outcome.HIT.value, Outcome.MISS.value]

    if isinstance(outcomes, list) and isinstance(values, list):
        if vs[0].ndim == 1:
            for s in range(len(outcomes)):
                vs[s][np.isin(outcomes[s], bad_outcomes)] = np.nan
        elif vs[0].ndim == 2:
            for s in range(len(outcomes)):
                vs[s][np.isin(outcomes[s], bad_outcomes), :] = np.nan
        else:
            raise NotImplementedError(vs[0].ndim)

    elif isinstance(outcomes, np.ndarray) and isinstance(values, np.ndarray):
        bad_idxs = np.logical_or(outcomes == bad_outcomes[0], outcomes == bad_outcomes[1])
        if vs.ndim == 1:
            vs[bad_idxs] = np.nan
        elif vs.ndim == 2:
            vs[bad_idxs, :] = np.nan
        else:
            raise NotImplementedError(vs.ndim)
    else:
        raise NotImplementedError(f"Object types are {type(outcomes)} and {type(values)}")

    return vs


def batch_nan_stat(values: type[np.ndarray | list[np.ndarray]], batch_size: int = 10, stat: str = 'mean') -> np.ndarray:
    """
    Extract mean or standard deviation for sequential batches of value arrays.
    :param values: Array (1D or 2D) or list of 1D-arrays of values to process
    :param batch_size: Size of the batches to make
    :param stat: 'mean' or 'std' to specify the statistic to extract
    :return: 1D or 2D array of the batch-dependent statistics (mean or std)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if isinstance(values, list):
            n_batches = len(values[0]) // batch_size
            batch_stat = np.zeros(n_batches)
            for b in range(n_batches):
                if stat == 'mean':
                    batch_stat[b] = np.nanmean(np.array([v[b*batch_size:(b+1)*batch_size] for v in values]))
                elif stat == 'std':
                    batch_stat[b] = np.nanstd(np.array([np.nanmean(v[b*batch_size:(b+1)*batch_size]) for v in values]))
                else:
                    raise NotImplementedError(stat)
        elif isinstance(values, np.ndarray):
            if values.ndim == 2:
                n_batches = values.shape[1] // batch_size
                batch_stat = np.zeros(n_batches)
                for b in range(n_batches):
                    if stat == 'mean':
                        batch_stat[b] = np.nanmean(values[:, b * batch_size:(b + 1) * batch_size])
                    elif stat == 'std':
                        batch_stat[b] = np.nanstd(np.nanmean(values[:, b * batch_size:(b + 1) * batch_size], axis=0))
                    else:
                        raise NotImplementedError(stat)
            elif values.ndim == 1:
                n_batches = values.shape[0] // batch_size
                batch_stat = np.zeros(n_batches)
                for b in range(n_batches):
                    if stat == 'mean':
                        batch_stat[b] = np.nanmean(values[b * batch_size:(b + 1) * batch_size])
                    elif stat == 'std':
                        batch_stat[b] = np.nanstd(values[b * batch_size:(b + 1) * batch_size])
                    else:
                        raise NotImplementedError(stat)
            else:
                raise NotImplementedError(values.ndim)
        else:
            raise NotImplementedError(type(values))

    return batch_stat


def get_significance(p_value: float) -> str:
    """
    Get significance stars for a p-value.
    :param p_value: P-value.
    """
    if p_value > 0.05 or np.isnan(p_value):
        s = 'n.s.'
    elif p_value > 0.01:
        s = '*'
    elif p_value > 0.001:
        s = '**'
    else:
        s = '***'
    return s


def set_style() -> None:
    """
    Set the plotting style, by defining some specific features.
    """
    matplotlib.rc(group='font', family='sans-serif')
    matplotlib.rc(group='font', serif='Arial')
    matplotlib.rc(group='text', usetex='false')
    matplotlib.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['font.size'] = FONT_SIZE
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['xtick.major.size'] = 2.
    matplotlib.rcParams['xtick.major.width'] = 0.5
    matplotlib.rcParams['xtick.minor.size'] = 1.2
    matplotlib.rcParams['xtick.minor.width'] = 0.35
    matplotlib.rcParams['ytick.major.size'] = matplotlib.rcParams['xtick.major.size']
    matplotlib.rcParams['ytick.major.width'] = matplotlib.rcParams['xtick.major.width']
    matplotlib.rcParams['ytick.minor.size'] = matplotlib.rcParams['xtick.minor.size']
    matplotlib.rcParams['ytick.minor.width'] = matplotlib.rcParams['xtick.minor.width']
    matplotlib.rcParams['axes.linewidth'] = AXIS_WIDTH
    matplotlib.rcParams['lines.linewidth'] = LINE_WIDTH
    matplotlib.rcParams['patch.linewidth'] = LINE_WIDTH


def adjust_figure(fig: plt.Figure, w_space: float = None, h_space: float = None) -> None:
    """
    Adjust the layout of the figure.
    :param fig: figure object
    :param w_space: width reserved for space between plots
    :param h_space: height reserved for space between plots
    """
    fig_width, fig_height = fig.get_size_inches()
    bottom = FIG_H_PAD / fig_height
    left = FIG_W_PAD / fig_width
    rect = (left, bottom, 1. - left, 1. - bottom - PANEL_LABEL_HEIGHT / fig_height)
    plt.tight_layout(pad=0., h_pad=0, w_pad=0, rect=rect)
    fig.subplots_adjust(wspace=w_space, hspace=h_space)


def save_or_show(saving: bool, plot_dir: Path, plot_name: str, plot_dpi=600) -> None:
    """
    Save the figure as svg and as pdf or just display the figure with the desired DPI.
    :param saving: Whether to save the figure. If false the figure will just be displayed.
    :param plot_dir: Directory where to save the figure
    :param plot_name: File name under which to save the figure (without suffix)
    :param plot_dpi: DPI to display the figure with
    """

    if saving:
        (plot_dir / SVG_DIR).mkdir(parents=True, exist_ok=True)
        matplotlib.pyplot.savefig(plot_dir / SVG_DIR / f"{plot_name}.svg")
        matplotlib.pyplot.savefig(plot_dir / f"{plot_name}.pdf", dpi=DPI)
        matplotlib.pyplot.close()
    else:
        matplotlib.pyplot.gcf().set_dpi(plot_dpi)
        matplotlib.pyplot.show()


def get_parula_cm() -> matplotlib.colors.ListedColormap:
    """
    Get the parula colormap, which is the default for matlab plots
    :return: The parula colormap
    """
    return matplotlib.colors.ListedColormap(PARULA, name='parula')


def get_performance_cm(reverse: bool = False) -> matplotlib.colors.LinearSegmentedColormap:
    """
    Get the colormap used to depict the evolution of the learning performance
    :param reverse: whether to reverse the colormap
    :return: The performance colormap
    """
    if reverse:
        return matplotlib.colors.LinearSegmentedColormap.from_list(name="", colors=['lime', 'green', 'silver', 'k'])
    else:
        return matplotlib.colors.LinearSegmentedColormap.from_list(name="", colors=['k', 'silver', 'green', 'lime'])


def plot_trace(x, y, color="b", alpha=0.5, label=None, ax=None, z=0, line_style='-') -> None:
    """
    Plot traces either of: (1) mean trace and area showing the standard deviation of the trace or (2) just a trace of
    singular values. Which of both is determined by the type of y. If y is a dict, the
    :param x: array of x-values
    :param y: array of y-values or dictionary with keys 'mean' and 'std' each containing an array
    :param color: color of trace and std area
    :param alpha: transparency of std area
    :param label: label for legend
    :param ax: axis object to plot on
    :param z: zorder of the trace
    :param line_style: style of the line
    """
    ls = {'color': color, 'label': label, 'zorder': z, 'linestyle': line_style}
    if type(y) is dict:
        idxs = np.isfinite(y["mean"])
        y1 = y["mean"] - y["std"]
        y2 = y["mean"] + y["std"]
        if ax is None:
            plt.fill_between(x[idxs], y1[idxs], y2[idxs], color=color, alpha=alpha)
        else:
            ax.fill_between(x[idxs], y1[idxs], y2[idxs], color=color, alpha=alpha)
        if ax is None:
            plt.plot(x[idxs], y["mean"][idxs], **ls)
        else:
            ax.plot(x[idxs], y["mean"][idxs], **ls)
    else:
        idxs = np.isfinite(y)
        if ax is None:
            plt.plot(x[idxs], y[idxs], **ls)
        else:
            ax.plot(x[idxs], y[idxs], **ls)


def plot_t(ax: plt.Axes, y: float, iti: float = 0, end: float = END_T) -> None:
    """
    Plot important time windows of the trial as colored sections of a horizontal bar
    :param ax: axis object to plot on
    :param y: y-axis height at which to plot the bar
    :param iti: Time (in HZ freq) when the inter-trial interval bar to plot begins
    :param end: Time (in HZ freq) when the outcome bar ends
    """
    ls = {'linewidth': 4, 'clip_on': False, 'zorder': 0, 'solid_capstyle': 'butt'}
    ax.plot([iti * HZ, TONE_T], [y, y], COL_ITI, **ls)
    ax.plot([TONE_T, TEXTURE_T], [y, y], COL_PRE, **ls)
    ax.plot([TEXTURE_T, OUTCOME_T - 1], [y, y], COL_TEXTURE, **ls)
    ax.plot([OUTCOME_T - 1, end], [y, y], COL_OUTCOME, **ls)


def plot_inhibition_bar(ax: plt.Axes, y: float, n_trials: int, inhibited: bool, height: float = 0.03) -> None:
    """
    Draw the apical inhibition protocol under the traces of a panel as in fig. 4h: a bar over the simulated trials,
    filled in green over the trials during which the apical dendrites are inhibited
    :param ax: axis object to plot on
    :param y: lower edge of the bar (in axes coordinates)
    :param n_trials: number of simulated trials
    :param inhibited: whether the apical dendrites are inhibited for the first N_TRIALS trials
    :param height: height of the bar (in axes coordinates)
    """
    style = {'transform': ax.get_xaxis_transform(), 'clip_on': False, 'zorder': 2}
    if inhibited:
        ax.add_patch(matplotlib.patches.Rectangle((0, y), N_TRIALS, height, facecolor=COL_RESTORED, linewidth=0,
                                                  **style))
    ax.add_patch(matplotlib.patches.Rectangle((0, y), n_trials, height, facecolor='none', linewidth=0.7,
                                              edgecolor=COL_RESTORED if inhibited else 'k', **style))


def plot_selectivity_histogram(ax: plt.Axes, selectivities: np.ndarray) -> np.ndarray:
    """
    Histogram of the texture selectivities of the texture-responsive neurons in bins of 0.2 from -MAX_GAIN to MAX_GAIN
    (the range of the selectivities once the apical gain is maximal; values outside it are clipped into it so that every
    neuron is counted), with the bins colored by selectivity class (fig. S12d)
    :param ax: axis object to plot on
    :param selectivities: texture selectivities of the texture-responsive neurons
    :return: numbers of non-selective, go-selective and no-go-selective neurons
    """
    bin_edges = np.linspace(-MAX_GAIN, MAX_GAIN, int(round(2 * MAX_GAIN / 0.2)) + 1)
    n_bins = len(bin_edges) - 1
    nb2 = int(n_bins / 2) - 1
    unselective_bins = [nb2, nb2+1]
    s2_bins = list(range(nb2))
    s1_bins = list(range(nb2 + 2, n_bins))
    data = np.clip(selectivities, bin_edges[0] + 0.1, bin_edges[-1] - 0.1)  # Count every neuron
    counts, _, patches = ax.hist(data, bins=bin_edges)
    for b in s1_bins:
        patches[b].set_facecolor(COL_T1)
    for b in s2_bins:
        patches[b].set_facecolor(COL_T2)
    for b in unselective_bins:
        patches[b].set_facecolor(COL_DISTRACTOR)
    ax.set_xlim([-MAX_GAIN, MAX_GAIN])
    return np.array([counts[unselective_bins].sum(), counts[s1_bins].sum(), counts[s2_bins].sum()])


def plot_perf_eg(saving: bool = True, mean: bool = True, sim: Simulation = Simulation.DEFAULT,
                 save_name: str = 'perf_example', fast: bool = False):
    """
    Performance trace example for the default model
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param mean: boolean encoding whether to plot an example or the mean performance
    :param sim: simulation identifier
    :param save_name: name under which to save the plot
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    plt.plot([0, N_TRIALS], [0.5, 0.5], "k:")
    trials = np.arange(N_TRIALS+1)
    results = get_results(simulation=sim, fast=fast)
    if mean:
        perfs = np.zeros((len(results), len(trials)))
        for seed in range(N_SEEDS):
            perfs[seed, :] = results[seed][K_PERFORMANCE]
        mean = np.mean(perfs, axis=0)
        std = np.std(perfs, axis=0)
        ax.fill_between(trials, mean - std, mean + std, clip_on=False, color='k', alpha=0.5, lw=0)
        ax.plot(trials, mean, 'k')
        xp_t = sum([results[seed][K_EXPERT_T] for seed in range(N_SEEDS)]) / N_SEEDS

    else:
        seed = 0
        ax.plot(trials, results[seed][K_PERFORMANCE], 'k')
        xp_t = results[seed][K_EXPERT_T]
    ax.plot([xp_t, xp_t], [0, 1], 'r--', lw=0.5, zorder=0)
    plt.xlim([0, N_TRIALS])
    plt.ylim([0, 1])
    plt.ylabel("Performance")
    plt.xlabel("Trial")

    # Performance evolution colormap
    p_cmap = get_performance_cm()
    cb_width = 0.05
    c_map_ax = ax.inset_axes([0., 0, 1, cb_width], zorder=-10)
    cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='horizontal')
    cb.outline.set_visible(False)
    cb.set_ticks([])

    # Add the patch to the Axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=save_name)


def plot_v_hat(saving=True, outcome_types: tuple[Outcome, ...] = (Outcome.HIT, Outcome.CR),
               save_name: str = 'v_hat_dynamics', fast: bool = False) -> None:
    """
    Dynamic of the state value estimate (V) during trials of specific outcome types for expert agents and the
    corresponding dynamic changes of the estimate (Delta V) and unsigned changes of the estimate (|Delta V|).
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param outcome_types: tuple of outcome type to plot.
    :param save_name: name under which to save the plot.
    :param fast: whether to plot the simulation of the fast-learning regime
    """

    # Get value estimates and trial outcomes
    results = get_results(fast=fast)
    outcomes = [results[s][K_OUTCOME] for s in range(N_SEEDS)]
    v_preds = [results[s][K_V_HAT] for s in range(N_SEEDS)]

    # Figure cosmetics
    n_outcomes = len(outcome_types)
    colors = {Outcome.HIT: COL_HIT, Outcome.CR: COL_CR, Outcome.FA: 'palevioletred', Outcome.MISS: 'olivedrab'}
    y_labels = ('Naïve\n$\widehat{V}$', 'Expert\n$\widehat{V}$', '$\Delta \widehat{V}_t$', '$|\Delta \widehat{V}_t|$')
    labels = {Outcome.HIT: 'Hit', Outcome.CR: 'CR', Outcome.FA: 'FA', Outcome.MISS: 'Miss'}

    # Define time steps to plot before the cue and after the outcome
    x = np.arange(END_T+1, dtype=int)
    simulated_x_ts = np.arange(TONE_T, END_T, dtype=int)

    # Define the trial windows for naive and expert learning stages
    trial_windows = (np.arange(5), np.arange(N_TRIALS - 50, N_TRIALS))

    # Variables to compute the largest y-axis range. This will be used to maintain the same y-scale across all plots
    bottom = 0.
    top = 0.
    y_diff = 0.

    # Init figure
    set_style()
    n_rows = 4 * n_outcomes + 1
    fig_size = (FIG_WIDTH / 4., 4 * n_outcomes / INCH)
    h_ratios = 4 * n_outcomes * [3] + [1]
    fig, axs = plt.subplots(nrows=n_rows, figsize=fig_size, sharex='all', height_ratios=h_ratios)

    # Plot the Hit case and the CR case
    for i in range(n_outcomes):

        # Plot the naive and expert learning stages of the state value estimate
        r0 = i * 4
        outcome_type = outcome_types[i]
        color = colors[outcome_type]
        v_preds_outcome = get_outcome_specific(outcome=outcome_type, outcomes=outcomes, values=v_preds)
        for j, window in enumerate(trial_windows):

            # Process the value estimate in the desired window of trials
            v_preds_outcome_window = np.array([v_preds_outcome[s][window, :] for s in range(N_SEEDS)])
            vp_mean, vp_ste = np.zeros(len(x)), np.zeros(len(x))
            vp_mean[simulated_x_ts] = np.nanmean(v_preds_outcome_window, axis=(0, 1))
            vp_ste[simulated_x_ts] = np.nanstd(v_preds_outcome_window, axis=(0, 1)) / np.sqrt(N_SEEDS)

            # Plot the traces with standard error
            axs[r0 + j].fill_between(x, vp_mean - vp_ste, vp_mean + vp_ste, color=color, alpha=0.3, linewidth=0.)
            axs[r0 + j].plot(x, vp_mean, color, label=labels[outcome_type])

            # Update the lowest and highest values to plot in the y-axis range
            bottom = min(bottom, np.min(vp_mean - vp_ste))
            top = max(top, np.max(vp_mean + vp_ste))
            y_diff = max(top - bottom, y_diff)

        # Compute the signed and unsigned change of state-value estimate during the trial and plot with bars
        v_preds_outcome_window = np.array([v_preds_outcome[s][trial_windows[1], :] for s in range(N_SEEDS)])
        dv_dt = np.diff(v_preds_outcome_window, axis=2)
        for j in (2, 3):

            # Process the data
            if j == 2:
                y = dv_dt
            else:
                y = np.abs(dv_dt)
            dvdt_mean = np.nanmean(y, axis=(0, 1))
            dvdt_ste = np.nanstd(y, axis=(0, 1)) / np.sqrt(N_SEEDS)

            # Bar plot
            axs[r0 + j].bar(
                x=simulated_x_ts[:-1] + 0.5, height=dvdt_mean, width=0.9, color=color, yerr=dvdt_ste, bottom=0.
            )

            # Update the lowest and highest values to plot in the y-axis range
            y_lim = axs[r0 + j].get_ylim()
            y_diff = max(y_lim[1] - y_lim[0], y_diff)

        # Cosmetics like removing axes, setting y-label and adding horizontal dashed line for zero
        for j in range(4):
            axs[r0 + j].plot([x[0], x[-1]], [0., 0.], 'k--', dashes=(5, 2), linewidth=AXIS_WIDTH, clip_on=False)
            axs[r0 + j].xaxis.set_visible(False)
            plt.setp(axs[r0 + j].spines.values(), visible=False)
            axs[r0 + j].tick_params(left=False, labelleft=False)
            axs[r0 + j].set_ylabel(y_labels[j], rotation='horizontal', verticalalignment='bottom', ha='center', y=0)

    # Adjust the y-axis range for the scale to be the same across all plots
    for row in range(n_rows-1):
        ax_bottom = min(axs[row].get_ylim()[0], -0.05)
        axs[row].set_ylim([ax_bottom, ax_bottom + y_diff])

    # Colored illustration of time windows
    plot_t(ax=axs[-1], y=0, end=x[-1])
    axs[-1].set_axis_off()

    # Finalize and save/show the figure
    txt_style = {'ha': 'right', 'va': 'center', 'fontsize': FONT_SIZE}
    for i, outcome_type in enumerate(outcome_types):
        ax = axs[i * 4]
        ax.text(x=0.95, y=0.3, s=labels[outcome_type], color=colors[outcome_type], transform=ax.transAxes, **txt_style)
    plt.suptitle('State-value estimate')
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=save_name, plot_dpi=150)


def plot_apical_raster(outcome: Outcome, expert_aligned: bool = False, fast: bool = False) -> None:
    """
    Raster plots of the evolution of responses for each apical dendrite type (sensory and outcome dendrites) of the
    initially go-selective and no-go-selective neurons (the outcome dendrites receive the same input in every neuron)
    during a specific trial type of choice.
    :param outcome: Outcome identifier specifying the trial type (Hit, Miss, CR or FA)
    :param expert_aligned: Whether to align trials to the first expert trial
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    rows = 2
    cols = 2
    batch_size = 20
    set_style()
    cmap = get_parula_cm()
    cmap.set_bad(color='k')
    cb_width = 0.041
    p_cmap = get_performance_cm(reverse=True)
    if expert_aligned:
        y_ticks = [-400, -200, 0]
        n_trials = N_BEFORE_EXPERT + N_EXPERTS
        t_min, t_max = 1 - n_trials + N_EXPERTS, N_EXPERTS
    else:
        y_ticks = [0, 500, 1000, 1500]
        n_trials = N_TRIALS
        t_min, t_max = 1, N_TRIALS
    n_batches = n_trials // batch_size
    results = get_results(fast=fast)
    selective = [get_texture_selective(result=r) for r in results]
    outcomes = get_trials(values=[r[K_OUTCOME] for r in results], results=results, expert_aligned=expert_aligned)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * PANEL_WIDTH, rows * PANEL_HEIGHT))
    for i in range(rows):
        for j in range(cols):
            if i == 0:
                values = [r[K_DENDRITE_OUTCOME] for r in results]
            else:
                values = [get_sensory_dendrites(result=r, neurons=sel[j]) for r, sel in zip(results, selective)]
            data = get_trials(values=values, results=results, expert_aligned=expert_aligned)
            outcome_spec_data = np.array(get_outcome_specific(outcome=outcome, outcomes=outcomes, values=data))
            apical = np.zeros((n_batches, N_TIME_STEPS))
            if i == 0:  # The outcome dendrites are only active at the outcome
                if batch_size > 1:
                    for b in range(n_batches):
                        apical[b, -1] = np.nanmean(outcome_spec_data[:, b * batch_size:(b + 1) * batch_size])
                else:
                    apical[:, -1] = np.nanmean(outcome_spec_data, axis=0)
            else:
                if batch_size > 1:
                    for b in range(n_batches):
                        apical[b, :] = np.nanmean(outcome_spec_data[:, b * batch_size:(b + 1) * batch_size, :],
                                                  axis=(0, 1))
                else:
                    apical = np.nanmean(outcome_spec_data, axis=0)
            img = np.pad(array=apical, pad_width=((0, 0), (HZ, 1)))
            axs[i, j].imshow(img, aspect='auto', extent=(0, END_T, t_max, t_min),
                             vmin=apical_transfer(0.), vmax=1., cmap=cmap)
            plot_t(ax=axs[i, j], y=t_max + (t_max-t_min) * 0.02)
            c_map_ax = axs[i, j].inset_axes([0., cb_width, cb_width, 1 - cb_width])
            cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='vertical')
            cb.outline.set_visible(False)
            cb.set_ticks([])
            for pos in ['right', 'top', 'bottom', 'left']:
                axs[i, j].spines[pos].set_visible(False)

            axs[i, j].set_ylim([t_max + (t_max-t_min) * 0.04, None])
            axs[i, j].set_xlim([-0.99, None])
            if i == 1:
                axs[i, j].set_xlabel('Trial time')
                axs[i, j].set_xticks([TONE_T + 0.5, TEXTURE_T + 0.5, OUTCOME_T + 0.5],
                                     ['Tone', 'Texture', 'Outcome'], zorder=10)
            else:
                axs[i, j].set_xticks([])
            if j == 0:
                axs[i, j].set_yticks(y_ticks)
                if i == 0:
                    axs[i, j].set_ylabel('Outcome Dendrite\n\nTrial ID')
                else:
                    axs[i, j].set_ylabel('Sensory Dendrite\n\nTrial ID')
            else:
                axs[i, j].set_yticks(y_ticks, [''] * len(y_ticks))
    axs[0, 0].title.set_text('Go-selective neurons')
    axs[0, 1].title.set_text('NoGo-selective neurons')
    adjust_figure(fig=fig, h_space=0.05)


def plot_apical_trace_boxplot(saving=True, verbose: bool = False, outcome: Outcome = None, y_lim: float = 1.01,
                              y_ticks: tuple[float, ...] = (0., 0.5, 1.), fast: bool = False) -> None:
    """
    Evolution of apical activity with learning during different trial windows and for each dendrite type as well as
    box-plots of activity depending on the learning phase and statistical significance between phases. The figure can be
    produced for specific trial outcome types (Hits, Misses, CRs or FAs) or for all trials (outcome=None). The sensory
    dendrites are those of the neurons that are texture-responsive in the first N_PHASE_TRIALS trials (as in a random
    sample of L5 neurons), the outcome dendrites receive the same input in every neuron.
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param verbose: whether to print out p-values
    :param outcome: the outcome type to plot the traces for. If None, all trial types are included.
    :param y_lim: upper limit of the apical activity axis of the traces
    :param y_ticks: ticks of the apical activity axis of the traces
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    # Initializations
    batch_size = 50
    p_cmap = get_performance_cm()
    cb_width = 0.05
    window_labels = ("Cue", "Touch", "Outcome")
    tcolors = (COL_PRE, COL_TEXTURE, COL_OUTCOME)
    window_ids = list(range(len(window_labels)))
    plot_order = list(reversed(window_ids))
    branch_names = ("Outcome dendrites", "Sensory dendrites")
    n_branches = len(branch_names)
    title_col = (COL_OUTCOME, COL_SENSORY)
    learning_phase_label = ('N', 'L', 'E')
    n_l_phase = len(learning_phase_label)
    n_seeds = N_SEEDS
    if outcome is None:
        plot_name = 'fig_4g_apical_responses'
        outcome_label = 'All trial types'
    elif outcome == Outcome.HIT:
        plot_name = 'fig_s13_apical_responses_hit'
        outcome_label = 'Hit trials'
    elif outcome == Outcome.CR:
        plot_name = 'fig_s13_apical_responses_cr'
        outcome_label = 'CR trials'
    elif outcome == Outcome.FA:
        plot_name = 'fig_s13_apical_responses_fa'
        outcome_label = 'FA trials'
    elif outcome == Outcome.MISS:
        plot_name = 'fig_s13x_apical_responses_miss'
        outcome_label = 'Miss trials'
    else:
        raise ValueError(outcome)

    # Initialize figure that will be split between traces (left) and boxplots (right)
    set_style()
    fig = plt.figure(constrained_layout=True, figsize=(0.5 * FIG_WIDTH, 8 / INCH))
    sub_figs = fig.subfigures(1, 2, width_ratios=[3, 1.5], wspace=0.07)
    axs = sub_figs[0].subplots(nrows=n_branches, ncols=1)

    # Get data for plotting (the sensory dendrites need the trials before the expert alignment)
    trials = np.zeros(0)
    rs = get_results(expert_aligned=False, fast=fast)
    # The traces show the N_BEFORE_EXPERT trials before the expert trial, or fewer (whole batches) if a seed becomes
    # expert earlier (in the fast-learning regime), so that every seed contributes to every batch of the traces
    n_before = min(N_BEFORE_EXPERT, batch_size * int(min(r[K_EXPERT_T] for r in rs) // batch_size))
    nt = min(N_TRIALS, n_before + N_EXPERTS)
    apical_sensory_all = [get_sensory_dendrites(result=r, neurons=get_texture_responsive(result=r)) for r in rs]
    apical_outcome_all = [r[K_DENDRITE_OUTCOME] for r in rs]
    outcomes_all = [r[K_OUTCOME] for r in rs]

    def get_window_activity(expert_aligned: bool) -> np.ndarray:
        if expert_aligned:
            select = [lambda v, t=int(r[K_EXPERT_T]): align_to_expert(values=v, expert_t=t, n_before=n_before)
                      for r in rs]
        else:
            select = [lambda v: v for _ in rs]
        apical_sensory = [select[s](apical_sensory_all[s]) for s in range(n_seeds)]
        apical_outcome = [select[s](apical_outcome_all[s]) for s in range(n_seeds)]
        if outcome is not None:
            outcomes = [select[s](outcomes_all[s]) for s in range(n_seeds)]
            apical_sensory = get_outcome_specific(outcome=outcome, outcomes=outcomes, values=apical_sensory)
            apical_outcome = get_outcome_specific(outcome=outcome, outcomes=outcomes, values=apical_outcome)
        y_out = np.array([apical_transfer(torch.zeros(n_seeds, len(apical_outcome[0]))).numpy(),
                          apical_transfer(torch.zeros(n_seeds, len(apical_outcome[0]))).numpy(),
                          apical_outcome])
        y_sen = np.array([[apical_sensory[s][:, 0] for s in range(n_seeds)],
                          [apical_sensory[s][:, TEXTURE_T - TONE_T] for s in range(n_seeds)],
                          [apical_sensory[s][:, -1] for s in range(n_seeds)]])
        return np.stack([y_out, y_sen], axis=0)

    ys = get_window_activity(expert_aligned=True)

    # Plot traces with standard error
    for dendrite_type in range(n_branches):
        for trial_window in window_ids:
            y = {"mean": batch_nan_stat(ys[dendrite_type, trial_window, :, :], batch_size=batch_size),
                 "std": batch_nan_stat(ys[dendrite_type, trial_window, :, :], batch_size=batch_size, stat='std') / np.sqrt(N_SEEDS)}
            trials = np.linspace(start=N_EXPERTS-nt, stop=N_EXPERTS, num=len(y['mean']), endpoint=False)
            plot_trace(
                trials, y, color=tcolors[trial_window], label=window_labels[trial_window], ax=axs[dendrite_type], alpha=0.2
            )

    # Figure cosmetics
    for dendrite_type in range(n_branches):

        # Colorbar of learning progression
        c_map_ax = axs[dendrite_type].inset_axes([0, 0., 1, cb_width])
        cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='horizontal')
        cb.outline.set_visible(False)
        cb.set_ticks([])

        # Expert line
        axs[dendrite_type].vlines(x=0, ymin=0, ymax=y_ticks[-1], color='k', linestyle='--', linewidth=AXIS_WIDTH)

        # Axis and labeling
        axs[dendrite_type].set_xlim([trials[0], trials[-1]])
        axs[dendrite_type].set_ylim([-0.05, y_lim])
        axs[dendrite_type].spines[['top', 'right']].set_visible(False)
        step = 100 if n_before > 200 else 50  # Every other tick labeled, and the last one
        x_ticks = np.arange(-step * (n_before // step), N_EXPERTS, step)
        axs[dendrite_type].set_xticks(x_ticks, [f'{t}' if t % (2 * step) == 0 or t == x_ticks[-1] else ''
                                                for t in x_ticks])
        axs[dendrite_type].set_yticks(y_ticks)
        axs[dendrite_type].set_yticklabels(y_ticks)
        axs[dendrite_type].spines['left'].set_bounds(0, y_ticks[-1])
        axs[dendrite_type].set_ylabel('Apical activity')
        axs[dendrite_type].set_title(branch_names[dendrite_type], color=title_col[dendrite_type])
        if dendrite_type == n_branches - 1:
            axs[-1].set_xlabel("Trial ID")
            axs[1].legend(loc='upper left', frameon=False)
        else:
            axs[dendrite_type].set_xticklabels([])
            axs[dendrite_type].text(0 - 0.03 * (trials[-1] - trials[0]), 0.95 * y_ticks[-1], outcome_label, ha='right')

    # Get the unaligned data
    ys = get_window_activity(expert_aligned=False)

    # Init the boxplot subfigure
    sub_sub_figs = sub_figs[1].subfigures(2, 1)
    n_trial_windows = 3
    boxplot_style = {'showfliers': False,
                     'widths': 0.4,
                     'medianprops': {'color': 'k', 'linewidth': AXIS_WIDTH, 'solid_capstyle': 'butt'},
                     'boxprops': {'linewidth': 0, 'facecolor': 'k'},
                     'whiskerprops': {'clip_on': False, 'linewidth': AXIS_WIDTH},
                     'capprops': {'clip_on': False, 'linewidth': AXIS_WIDTH}}

    # Init statistics variables
    idx_pair = [[0, 1], [1, 2], [0, 2]]
    str_pair = ['naïve vs. learn', 'learn vs. expert', 'naïve vs. expert']
    stat_style = {'va': 'center', 'ha': 'center', 'color': 'k', 'fontsize': FONT_SIZE-2, 'weight': 'bold'}
    stat_pair_ids = [0, 1, 2]

    # Plot the boxplots and statistical significance
    warnings.simplefilter("ignore")
    for dendrite_type in range(n_branches):
        sub_sub_figs[dendrite_type].text(-0.07, 0.5, 'Apical activity', va='center', rotation='vertical')
        axs = sub_sub_figs[dendrite_type].subplots(nrows=n_trial_windows, ncols=1)
        for ax_i, trial_window in enumerate(plot_order):

            # Get data by learning phase
            data = ys[dendrite_type, trial_window, :, :]
            ys_by_phase = [np.zeros(n_seeds) for _ in range(n_l_phase)]
            for seed in range(n_seeds):
                ys_by_phase[0][seed] = np.nanmean(data[seed, :rs[seed][K_LEARNING_T]])
                ys_by_phase[1][seed] = np.nanmean(data[seed, rs[seed][K_LEARNING_T]:rs[seed][K_EXPERT_T]])
                ys_by_phase[2][seed] = np.nanmean(data[seed, rs[seed][K_EXPERT_T]:])

            # Clean data (sometimes empty learning phase led to NaNs)
            for p in range(n_l_phase):
                ys_by_phase[p] = ys_by_phase[p][~np.isnan(ys_by_phase[p])]

            # Plot the boxplots
            boxplot_style['boxprops']['facecolor'] = tcolors[trial_window]
            axs[ax_i].boxplot(ys_by_phase, positions=stat_pair_ids, patch_artist=True, **boxplot_style)

            # Run the Tuckey HSD statistical test (not if a learning phase has fewer than two seeds, as in the
            # fast-learning regime, where the learning trial is the expert trial in most seeds)
            if verbose:
                print(f'{branch_names[dendrite_type]} during the {window_labels[trial_window]} window:')
            try:
                if min(len(y) for y in ys_by_phase) < 2:
                    raise RuntimeWarning('Fewer than two seeds in a learning phase')
                p_values = stats.tukey_hsd(*tuple(ys_by_phase)).pvalue
            except RuntimeWarning:
                p_values = np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])

            # Plot significance stars
            for k in stat_pair_ids:
                s = get_significance(p_value=p_values[idx_pair[k][0], idx_pair[k][1]])
                if s != get_significance(1.):
                    sx = sum(idx_pair[k]) / 2.
                    sy = 1.2 + 0.3 * k
                    axs[ax_i].hlines(y=sy, xmin=idx_pair[k][0], xmax=idx_pair[k][1], color='k', clip_on=False, linewidth=AXIS_WIDTH)
                    axs[ax_i].text(sx, sy + 0.04, s, **stat_style)
                if verbose:
                    print(f'  - {str_pair[k]} {s} (p-value={p_values[idx_pair[k][0], idx_pair[k][1]]:.1e})')

            # Axes cosmetics
            axs[ax_i].set_xlim((-0.4, 2.4))
            axs[ax_i].set_ylim((0, 2))
            axs[ax_i].spines[['top', 'right']].set_visible(False)
            axs[ax_i].tick_params(axis=u'x', which=u'both', length=0)
            axs[ax_i].set_yticks([0, 1., 2.], ['0', '', '2'])
            ax_right = axs[ax_i].twinx()
            ax_right.spines[['top', 'right']].set_visible(False)
            ax_right.set_yticks([])
            ax_right.set_ylabel(window_labels[trial_window], color=tcolors[trial_window], labelpad=7, ha='center', va='center')
            if ax_i == n_trial_windows - 1:
                axs[ax_i].set_xticklabels(['N', 'L', 'E'])
            else:
                axs[ax_i].set_xticklabels([])
    warnings.simplefilter("default")

    # Save or show figure
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_dpi=300, plot_name=plot_name)


def f4c_performance_eg(saving: bool = True, fast: bool = False) -> None:
    """
    Plot an example performance trace for the default simulation
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    plot_perf_eg(saving=saving, mean=False, sim=Simulation.DEFAULT, save_name='fig_4c_performance_example', fast=fast)


def f4d_expert_v_pred(saving=True, fast: bool = False) -> None:
    """
    Dynamic of the state value estimate (V) during Hit and CR trials for expert agents and the
    corresponding dynamic changes of the estimate (Delta V) and unsigned changes of the estimate (|Delta V|).
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    plot_v_hat(saving=saving, outcome_types=(Outcome.HIT, Outcome.CR), save_name='fig_4d_expert_v_pred', fast=fast)


def f4e_apical_dendrites_raster_plots(saving: bool = True, fast: bool = False) -> None:
    """
    Raster plots of the evolution of responses for each apical dendrite type (sensory and outcome dendrites) of the
    initially go-selective and no-go-selective neurons during either Hit or CR trials.
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    plot_apical_raster(outcome=Outcome.HIT, fast=fast)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name='fig_4e_apical_hit', plot_dpi=300)

    plot_apical_raster(outcome=Outcome.CR, fast=fast)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name='fig_4e_apical_cr', plot_dpi=300)


def f4e_soma_raster_plots(saving: bool = True, expert_aligned: bool = False, fast: bool = False):
    """
    Raster plots of the evolution of the mean somatic activity in the texture window of the initially go-selective and
    of the initially no-go-selective neurons during each type of trial (Hit, CR, FA, Miss).
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param expert_aligned: Whether to align trials to the first expert trial
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    rows = 2
    cols = 2
    batch_size = 50
    if expert_aligned:
        y_ticks = [-400, -200, 0]
        n_trials = N_BEFORE_EXPERT + N_EXPERTS
        t_min, t_max = 1 - n_trials + N_EXPERTS, N_EXPERTS
    else:
        y_ticks = [0, 500, 1000, 1500]
        n_trials = N_TRIALS
        t_min, t_max = 1, N_TRIALS
    n_batches = n_trials // batch_size
    set_style()
    cmap = get_parula_cm()
    cmap.set_bad(color='k')
    cb_width = 0.041
    p_cmap = get_performance_cm(reverse=True)
    results = get_results(fast=fast)
    selective = [get_texture_selective(result=r) for r in results]
    outcomes = get_trials(values=[r[K_OUTCOME] for r in results], results=results, expert_aligned=expert_aligned)
    outcome_types = [[Outcome.HIT, Outcome.MISS], [Outcome.FA, Outcome.CR]]
    for k in range(2):
        data = get_trials(values=[np.mean(r[K_X_SOM_TXT][:, sel[k]], axis=1) for r, sel in zip(results, selective)],
                          results=results, expert_aligned=expert_aligned)
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * PANEL_WIDTH, rows * PANEL_HEIGHT))
        for i in range(rows):
            for j in range(cols):
                out_spec_data = np.abs(np.array(get_outcome_specific(outcome=outcome_types[i][j], outcomes=outcomes,
                                                                     values=data)))
                mean_data = np.zeros((n_batches, END_T))
                if batch_size > 1:
                    for b in range(n_batches):
                        mean_data[b, TEXTURE_T] = np.nanmean(out_spec_data[:, b * batch_size:(b + 1) * batch_size])
                else:
                    mean_data = np.nanmean(out_spec_data, axis=0)
                mean_data[np.isnan(mean_data)] = 0
                axs[i, j].imshow(mean_data, aspect='auto', extent=(0, END_T, t_max, t_min), vmin=0.,
                                 vmax=5., cmap=cmap)
                plot_t(ax=axs[i, j], y=t_max + (t_max-t_min) * 0.02)
                c_map_ax = axs[i, j].inset_axes([0., cb_width, cb_width, 1 - cb_width])
                cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='vertical')
                cb.outline.set_visible(False)
                cb.set_ticks([])
                for pos in ['right', 'top', 'bottom', 'left']:
                    axs[i, j].spines[pos].set_visible(False)
                axs[i, j].set_ylim([t_max + (t_max-t_min) * 0.04, None])
                axs[i, j].set_xlim([-0.99, None])
                if i == 1:
                    axs[i, j].set_xlabel('Trial time')
                    axs[i, j].set_xticks([TONE_T + 0.5, TEXTURE_T + 0.5, OUTCOME_T + 0.5],
                                         ['Tone', 'Texture', 'Outcome'], zorder=10)
                else:
                    axs[i, j].set_xticks([])
                if j == 0:
                    axs[i, j].set_yticks(y_ticks)
                    if i == 0:
                        axs[i, j].set_ylabel('Go-selective neurons\n\nTrial ID')
                    else:
                        axs[i, j].set_ylabel('NoGo-selective neurons\n\nTrial ID')
                else:
                    axs[i, j].set_yticks(y_ticks, [''] * len(y_ticks))
        axs[0, 0].title.set_text('Lick')
        axs[0, 1].title.set_text('No Lick')

        adjust_figure(fig=fig, h_space=0.05)
        save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=f'fig_4e_t{k+1}_soma', plot_dpi=300)


def f4f_sen_dendrite(saving: bool = True, expert_aligned: bool = False, fast: bool = False) -> None:
    """
    Multi-panel plot showing:
    - Evolution of the sensory dendrite synaptic weight strengths depending on the bottom-up (basal) selectivity of the
      pyramidal neurons (with learning): the initially cue-, go- and no-go-selective neurons and all the other neurons
    - Evolution of the multiplicative gains of different stimuli with learning (for the selective neurons at the time of
      their stimulus, for the others at the time of their strongest basal input)
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param expert_aligned: Whether to align trials to the first expert trial
    :param fast: whether to plot the simulation of the fast-learning regime
    """

    # Initializations and loading data
    if expert_aligned:
        n_trials = N_BEFORE_EXPERT + N_EXPERTS
        t_min, t_max = 1 - n_trials + N_EXPERTS, N_EXPERTS
    else:
        n_trials = N_TRIALS
        t_min, t_max = 1, N_TRIALS
    results = get_results(fast=fast)
    batch_size = 50
    n_batches = n_trials // batch_size
    set_style()
    p_cmap = get_performance_cm(reverse=True)
    cb_width = 0.043

    # Figure with the learning progression as a colorbar along the trial axis of each panel
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(PANEL_WIDTH, 8 / INCH), sharey='all')
    for i in range(2):
        c_map_ax = axs[i].inset_axes([-cb_width, 0, cb_width, 1], zorder=-10)
        cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='vertical')
        cb.outline.set_visible(False)
        cb.set_ticks([])
        cb.set_label('Trial ID', labelpad=-15)
        cb.ax.tick_params(size=0)

    # Plot traces of the sensory dendrite synapse strengths and of the pyramidal neuron gains
    trials = np.linspace(t_min, t_max, n_batches, endpoint=True)
    plot_cols = [COL_PRE, COL_T1, COL_T2, COL_DISTRACTOR]
    keys = [K_W_AP, K_GAIN]
    x_ticks = [[0, 0.5, 1], [0, 2, 4, 6, 8, 10]]
    x_labs = [r'Apical synaptic weight $w^\mathrm{ap}$', r'Apical gain $g$']
    labels = ['Cue-selective', 'Go-selective', 'NoGo-selective', 'Other neurons']
    groups = []  # The initially cue-, go- and no-go-selective neurons and all the other neurons of each seed
    for r in results:
        cue = get_cue_selective(result=r)
        go, nogo = get_texture_selective(result=r)
        groups.append([cue, go, nogo, ~(cue | go | nogo)])
    time_steps = [0, TEXTURE_T - TONE_T, TEXTURE_T - TONE_T]  # Time steps of the stimulus each group is selective for
    for row in range(2):
        mean_data = np.zeros((n_batches, 4))
        for i in range(4):
            if keys[row] == K_GAIN and i < 3:  # The selective neurons while their stimulus is present
                values = [get_gains(result=r, neurons=g[i], time_step=time_steps[i]) for r, g in zip(results, groups)]
            else:  # The recorded gains, at the time step of the strongest initial basal input of each neuron
                values = [r[keys[row]][:, g[i]] for r, g in zip(results, groups)]
            data = np.concatenate(get_trials(values=values, results=results, expert_aligned=expert_aligned), axis=1)
            st_err = np.zeros(n_batches)
            for b in range(n_batches):
                mean_data[b, i] = np.nanmean(data[b * batch_size:(b + 1) * batch_size])
                st_err[b] = np.nanstd(data[b * batch_size:(b + 1) * batch_size]) / np.sqrt(batch_size)
            axs[row].fill_betweenx(trials, mean_data[:, i] - st_err, mean_data[:, i] + st_err,
                                          color=plot_cols[i], alpha=0.3, linewidth=0.)
        for i in range(4):
            axs[row].plot(mean_data[:, i], trials, plot_cols[i], label=labels[i])
        axs[row].spines[['right', 'top', 'left']].set_visible(False)
        axs[row].set_xlabel(x_labs[row])
        axs[row].set_xticks(x_ticks[row])
        axs[row].set_xlim([x_ticks[row][0], x_ticks[row][-1]])
        axs[row].set_yticks([])
        axs[row].set_ylim([t_min, t_max])
        axs[row].invert_yaxis()

    axs[0].legend(frameon=False, bbox_to_anchor=(1., 1.1), loc='upper right')
    adjust_figure(fig=fig, h_space=0.5)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=f'fig_4f_wap_g', plot_dpi=250)


def f4g_apical_window_traces(saving=True, verbose: bool = False, fast: bool = False) -> None:
    """
    Evolution of apical activity with learning during different trial windows and for each dendrite type (the sensory
    dendrites of the neurons that are texture-responsive in the first trials).
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param verbose: whether to print out p-values
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    plot_apical_trace_boxplot(saving=saving, verbose=verbose, y_lim=0.9, y_ticks=(0., 0.4, 0.8), fast=fast)


def plot_inhibition_performance(saving: bool, plot_name: str, simulation: Simulation = Simulation.DEFAULT,
                                fast: bool = False, plot_dir: Path = None) -> None:
    """
    Performance traces of each seed without apical inhibition (grey) and with the apical dendrites inhibited for the
    first N_TRIALS trials (green), as in fig. 4h
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param simulation: Simulation.DEFAULT for the model without bottom-up plasticity (Simulation.APICAL_INHIBITION in
    the inhibition protocol) or a bottom-up plasticity simulation (both protocols of its rule)
    :param fast: whether to plot the simulations of the fast-learning regime
    :param plot_dir: directory in which to save the plot (None: that of the learning regime, see get_plot_dir)
    """
    n_plot_seeds = N_SEEDS
    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    max_t = [N_TRIALS, N_TRIALS_AP_INH]
    plot_col = ["grey", COL_RESTORED]
    plt.plot([0, N_TRIALS_AP_INH], [0.5, 0.5], "k:")
    trials = np.arange(N_TRIALS_AP_INH)
    for sim in reversed(range(2)):  # The inhibition protocol first
        results = get_bu_results(simulation=simulation, inhibition=sim == 1, fast=fast)
        for s in range(n_plot_seeds):
            plt.plot(trials[:max_t[sim]], results[s][K_PERFORMANCE][:max_t[sim]],
                     plot_col[sim], zorder=0)
    plt.xlim([0, max_t[1]])
    y_min = 0.25
    plt.ylim([y_min, 1])
    plt.plot([0, N_TRIALS_AP_INH], [y_min + 0.07, y_min + 0.07], linewidth=3, color="w", zorder=1)
    plt.plot([0, N_TRIALS], [y_min + 0.03, y_min + 0.03], linewidth=3, color=COL_RESTORED, zorder=1)
    plt.plot([N_TRIALS, N_TRIALS_AP_INH], [y_min + 0.03, y_min + 0.03], linewidth=3, color="w", zorder=1)
    plt.ylabel("Performance")
    plt.xlabel("Trial")
    plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500], labels=[0, "", 1000, "", 2000, "", 3000, ""])
    plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=["", "", "0.6", "", "0.8", "", "1.0"])
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.85)

    plt.text(-0.02, 0., "Apical\ninhibition", transform=ax.transAxes, ha="right")

    # Add the patch to the Axes
    x0 = 1
    length_x = N_TRIALS_AP_INH - 2 * x0
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=(x0, y_min + 0.06), width=length_x, height=0.02, linewidth=0.7, edgecolor="k", facecolor='none',
        clip_on=False, zorder=2
    ))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=(x0, y_min + 0.02), width=length_x, height=0.02, linewidth=0.7, edgecolor=COL_RESTORED, facecolor='none',
        clip_on=False, zorder=2
    ))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_label_coords(-0.18, 0.6)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast) if plot_dir is None else plot_dir, plot_name=plot_name)


def f4h_performance(saving=True, fast: bool = False) -> None:
    """
    Performance traces with and without apical inhibition
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param fast: whether to plot the simulations of the fast-learning regime
    """
    plot_inhibition_performance(saving=saving, plot_name='fig_4h_performance', fast=fast)


def plot_inhibition_expert_trials(saving: bool, plot_name: str, simulation: Simulation = Simulation.DEFAULT,
                                  verbose: bool = False, fast: bool = False) -> None:
    """
    Distribution of number of trials necessary to reach expert performance depending on whether the apical dendrites
    were inhibited during the first 1800 trials or not, as in fig. 4h.
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param simulation: Simulation.DEFAULT for the model without bottom-up plasticity (Simulation.APICAL_INHIBITION in
    the inhibition protocol) or a bottom-up plasticity simulation (both protocols of its rule)
    :param verbose: boolean encoding whether to print statistical tests between the number of trials to become expert.
    :param fast: whether to plot the simulations of the fast-learning regime
    """
    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, 0.7 * PANEL_HEIGHT))
    plot_col = ['k', COL_RESTORED]
    face_col = ['grey', 'white']
    plot_order = [2, 0]
    xp_ts: list[np.ndarray] = [np.zeros(0), np.zeros(0)]
    for sim in (1, 0):
        results = get_bu_results(simulation=simulation, inhibition=sim == 1, fast=fast)
        expert_t = [results[s][K_EXPERT_T] for s in range(N_SEEDS)]
        xp_ts[sim] = np.array(expert_t)
        plt.barh(plot_order[sim], np.nanmean(expert_t), capsize=3.5, linewidth=1., edgecolor=plot_col[sim],
                 color=(1, 1, 1, 0.), xerr=np.nanstd(expert_t), align='center', fc=face_col[sim])
        plt.plot(expert_t, [plot_order[sim]] * N_SEEDS, 'ko', alpha=1, markersize=1)
        if verbose:
            print(f'{np.nanmean(expert_t):.0f} ± {np.nanstd(expert_t):.0f}')
    plt.barh(1, np.nanmean(xp_ts[1]-1800), capsize=3.5, linewidth=1., edgecolor='k',
             color=(1, 1, 1, 0.), xerr=np.nanstd(xp_ts[1]-1800), align='center')
    plt.plot(xp_ts[1]-1800, [1] * N_SEEDS, 'ko', alpha=1, markersize=1)
    plt.vlines(x=N_TRIALS, ymin=-0.5, ymax=2.5)
    plt.xlabel("Trial")
    plt.xlim([0, N_TRIALS_AP_INH])
    plt.yticks([2, 1, 0], ["Unperturbed", "Corrected", "Perturbed"])

    p_values = [stats.ranksums(xp_ts[0], xp_ts[1])[1], stats.ranksums(xp_ts[0], xp_ts[1] - 1800)[1]]
    if verbose:
        print(f'Perturbed vs. Unperturbed {get_significance(p_values[0])} {p_values[0]:.1e}')
        print(f'Corrected vs. Unperturbed {get_significance(p_values[1])} {p_values[1]:.1e}')
    xs = [3500, 2500]
    for i in range(2):
        plt.plot([xs[i], xs[i]], [i, 2], 'k-')
        s = get_significance(p_values[i])
        if s == get_significance(1.):
            plt.text(xs[i] + 100, i + (2 - i)/2, s, rotation=270, va='center', ha='center')
        else:
            plt.text(xs[i] + 50, i + (2 - i)/2, s, rotation=270, va='center', ha='center', weight='bold')
    fig.tight_layout()

    ax = fig.get_axes()[0]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def f4h_expert_trials(saving=True, verbose: bool = False, fast: bool = False) -> None:
    """
    Distribution of number of trials necessary to reach expert performance depending on whether the apical dendrites
    were inhibited during the first 1800 trials or not.
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param verbose: boolean encoding whether to print statistical tests between the number of trials to become expert.
    :param fast: whether to plot the simulations of the fast-learning regime
    """
    plot_inhibition_expert_trials(saving=saving, plot_name='fig_4h_expert_trials', verbose=verbose, fast=fast)


def plot_inhibition_w_ap(saving: bool, plot_name: str, simulation: Simulation = Simulation.DEFAULT,
                         fast: bool = False) -> None:
    """
    Evolution of the excitatory synaptic weights on the sensory dendrites with learning, without and with apical
    inhibition, as in fig. 4h: mean apical weight w_ap, in each seed, of the neurons that are texture-selective (go- or
    no-go-selective, solid lines) and of those that are texture-unresponsive (dashed lines) in the first N_PHASE_TRIALS
    trials of each protocol (the classes of fig. S12d)
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param simulation: Simulation.DEFAULT for the model without bottom-up plasticity (Simulation.APICAL_INHIBITION in
    the inhibition protocol) or a bottom-up plasticity simulation (both protocols of its rule)
    :param fast: whether to plot the simulations of the fast-learning regime
    """

    # Plot S1 top-down weight strength evolution
    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    plotcols = ['grey', COL_RESTORED]
    line_styles = ['-', '--']  # The texture-selective and the texture-unresponsive neurons
    for i in [1, 0]:
        results = get_bu_results(simulation=simulation, inhibition=i == 1, fast=fast)
        trials = np.arange(len(results[0][K_W_AP]))
        for r in results:
            go, nogo = get_texture_selective(result=r)
            for neurons, line_style in zip((go | nogo, ~get_texture_responsive(result=r)), line_styles):
                ax.plot(trials, np.mean(r[K_W_AP][:, neurons], axis=1, dtype=np.float64), color=plotcols[i],
                        linestyle=line_style, linewidth=0.5 * LINE_WIDTH, clip_on=False)

    # Plot style
    y_min = -0.25
    plt.ylim([y_min, 1])
    plt.xlim([0., N_TRIALS_AP_INH])
    plt.ylabel("Synapse strength $w^{\mathrm{ap}}$")
    plt.xlabel("Trial")
    plt.xticks([0, 1000, 2000, 3000, 4000], ['', '1000', '2000', '3000', ''])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    fig.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Apical inhibition bars
    x0 = 3
    y0 = y_min / 7.
    k = (y0 - y_min) / 7.
    x_length = N_TRIALS_AP_INH - 2 * x0
    y_width = 2 * k
    plt.plot([0, N_TRIALS], [y0 - 5 * k, y0 - 5 * k], linewidth=3.6, color=COL_RESTORED, zorder=0)
    ax.add_patch(matplotlib.patches.Rectangle(
        (x0, y0 - 6 * k), x_length, height=y_width, linewidth=0.7, edgecolor=COL_RESTORED, facecolor='none',
        clip_on=False
    ))
    ax.add_patch(matplotlib.patches.Rectangle(
        (x0, y0 - 3 * k), x_length, height=y_width, linewidth=0.7, edgecolor='k', facecolor='none', clip_on=False
    ))
    plt.text(-0.02, 0., "Apical\ninhibition", transform=ax.transAxes, ha="right")
    ax.yaxis.set_label_coords(-0.2, 0.6)

    adjust_figure(fig=fig)
    # Legend of the neuron groups in the margin above the axes (the traces leave no room for it inside), right-aligned
    # with the axes
    handles = [matplotlib.lines.Line2D([], [], color='k', linestyle=ls, linewidth=0.5 * LINE_WIDTH, label=label)
               for ls, label in zip(line_styles, ('Texture-selective', 'Texture-unresponsive'))]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(ax.get_position().x1, 1.), frameon=False,
               handlelength=2., borderaxespad=0., borderpad=0., labelspacing=0.2)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def f4h_w_ap(saving=True, fast: bool = False) -> None:
    """
    Evolution of the apical synaptic weights of the initially texture-selective and texture-unresponsive neurons with
    and without apical inhibition
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param fast: whether to plot the simulations of the fast-learning regime
    """
    plot_inhibition_w_ap(saving=saving, plot_name='fig_4h_w_ap', fast=fast)


def fs11bc_transfer_f(saving=True) -> None:
    """
    Transfer functions mapping apical excitation to apical activation, apical activation to multiplicative gain and the
    combination of both, which gives the multiplicative gain depending on the apical excitation.
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    set_style()
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(3 * PANEL_WIDTH, PANEL_HEIGHT))

    x0, x1 = -0.25, 1.25
    x = torch.tensor(np.linspace(x0, x1, 100))
    axs[0].plot(x, np.clip(apical_transfer(x), 0, 1), color="k", label=r'$\varphi (x)$')
    axs[0].set_ylim([-0.05, 1.01])
    axs[0].set_ylabel(r'Apical activity $x^\mathrm{ap}$')
    axs[0].set_xlabel("Apical input")
    axs[0].spines['left'].set_bounds(-0.05, 1)

    axs[1].plot(x, gain(x), color="k", label=r'$g(x)$')
    axs[1].set_ylim([0, None])
    axs[1].set_ylabel(r'Gain modulation $g$')
    axs[1].set_xlabel(r'Apical activity $x^\mathrm{ap}$')
    axs[1].spines['left'].set_bounds(0, MAX_GAIN)

    axs[2].plot(x, gain(apical_transfer(x)), color="k", label=r'$g(\varphi (x))$')
    axs[2].set_ylim([0, MAX_GAIN*1.01])
    axs[2].set_ylabel(r'Gain modulation $g$')
    axs[2].set_xlabel("Apical input")
    axs[2].spines['left'].set_bounds(0, MAX_GAIN)

    for c in range(3):
        axs[c].set_xlim([x0, x1])
        axs[c].spines['top'].set_visible(False)
        axs[c].spines['right'].set_visible(False)
        axs[c].legend(loc='lower right')

    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_dpi=300, plot_name='fig_s11bc_transfer_functions')


def fs11d_expert_v_pred(saving=True, fast: bool = False) -> None:
    """
    Dynamic of the state value estimate (V) during HFA trials for expert agents and the
    corresponding dynamic changes of the estimate (Delta V) and unsigned changes of the estimate (|Delta V|).
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    plot_v_hat(saving=saving, outcome_types=(Outcome.FA,), save_name='fig_s11d_expert_v_pred', fast=fast)


def fs11e_apical_dendrites_raster_plots(saving: bool = True, fast: bool = False) -> None:
    """
    Raster plots of the evolution of responses for each apical dendrite type (sensory and outcome dendrites) of the
    initially go-selective and no-go-selective neurons during FA trials.
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    plot_apical_raster(outcome=Outcome.FA, fast=fast)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name='fig_s11e_apical_fa', plot_dpi=300)


def fs12a_performance_eg(saving: bool = True, fast: bool = False) -> None:
    """
    Plot an example performance trace for the default simulation
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    plot_perf_eg(saving=saving, mean=False, sim=Simulation.DEFAULT, save_name='fig_s12a_performance_example', fast=fast)


def fs12b_wap_traces(saving: bool = False, fast: bool = False):
    """
    Traces of the sensory dendrite synapse strengths for pyramidal neurons depending on their bottom-up selectivity for
    the go-stimulus (which is encoded by the basal synaptic strength of the texture 1 input).
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    n_available_bins = N_AV_BIN
    s_idxs = [T1_IDX, T2_IDX]
    colors = [COL_T1, COL_T2]
    c_labels = ['$w^{bas}_{Go}$', '$w^{bas}_{NoGo}$']
    p_cmap = get_performance_cm()
    cb_width = 0.1
    set_style()
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    trials = torch.linspace(1, N_TRIALS, steps=N_TRIALS)
    results = get_results(simulation=Simulation.DEFAULT, fast=fast)
    for s in range(len(s_idxs)):
        w_aps = {}
        for result in results:
            w_bas = result[K_W_BAS]
            for i in range(w_bas.shape[1]):
                b = min(floor(w_bas[s_idxs[s], i] * n_available_bins), BIN_MAX)
                if b not in w_aps.keys():
                    w_aps[b] = torch.tensor(result[K_W_AP][:, i])[None, :]
                else:
                    w_aps[b] = torch.cat((w_aps[b], torch.tensor(result[K_W_AP][:, i])[None, :]), dim=0)
        n_bins = len(w_aps)

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [COL_DISTRACTOR, colors[s]])
        for b in range(n_bins):
            axs[s].plot(trials, torch.mean(w_aps[b], dim=0), color=cmap(b/(n_bins-1.)))
        axs[s].set_ylim([-cb_width, 1.1])
        axs[s].spines['left'].set_bounds(0, 1)
        axs[s].set_ylabel('$w^{ap}$')
        axs[s].spines[['right', 'top']].set_visible(False)
        axs[s].set_xlim([trials[0], trials[-1]])
        norm = matplotlib.colors.Normalize(vmin=0.5/n_bins, vmax=1-0.5/n_bins)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        ticks = np.linspace(0., 1, n_bins + 1, endpoint=True)
        cbar = plt.colorbar(mappable=sm, boundaries=ticks, ticks=ticks, ax=axs[s])
        tick_labels: list = [None for _ in range(n_bins + 1)]
        tick_labels[0] = 0
        tick_labels[-1] = n_bins / n_available_bins
        cbar.ax.set_yticklabels(tick_labels)
        cbar.ax.set_ylabel(c_labels[s], rotation=270, labelpad=-1)
        c_map_ax = axs[s].inset_axes([0., 0, 1, cb_width], zorder=-10)
        cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='horizontal')
        cb.outline.set_visible(False)
        cb.set_ticks([])
    axs[0].set_xticks([])
    axs[0].spines[['bottom']].set_visible(False)
    axs[1].set_xlabel('Trial')

    adjust_figure(fig=fig, h_space=0.3)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name='fig_s12b_w_ap_traces')


def plot_theta0_w_ap(saving: bool, plot_name: str, fast: bool = False, plot_dir: Path = None) -> None:
    """
    Sensory dendrite synapse strength after 1800 trials for pyramidal neurons depending on their bottom-up selectivity
    for the go-stimulus (which is encoded by the basal synaptic strength of the texture 1 input) and on the chosen
    value for the theta_0 parameter. The seeds whose simulation diverged are left out (in the fast-learning regime, at
    small theta_0), the values of theta_0 at which every seed diverged are shaded (lighter if only some did), and the
    value of theta_0 of the learning regime is marked by a dashed line.
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param fast: whether to plot the theta_0 sweep of the fast-learning regime
    :param plot_dir: directory in which to save the plot (None: that of the learning regime, see get_plot_dir)
    """
    n_available_bins = 20
    theta_0 = get_thetas_0(fast=fast)
    s_idxs = [T1_IDX, T2_IDX]
    ns = len(s_idxs)
    n_theta = len(theta_0)
    set_style()
    fig, axs = plt.subplots(nrows=1, ncols=ns, figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    n_bins = BIN_MAX + 1
    w_ap_final_mean = torch.full((ns, n_theta, n_available_bins), np.nan)  # nan where every seed diverged
    diverged, partly_diverged = [], []  # Values of theta_0 at which every seed or only some seeds diverged
    for t in range(n_theta):
        all_results = get_results(simulation=Simulation.DEFAULT, theta_0=theta_0[t], fast=fast)
        results = [r for r in all_results if not get_diverged(result=r)]
        if not results:
            diverged.append(theta_0[t])
        elif len(results) < len(all_results):
            partly_diverged.append(theta_0[t])
        for s in range(ns):
            w_ap_final = {}
            for result in results:
                w_bas = result[K_W_BAS]
                for i in range(w_bas.shape[1]):
                    b = min(floor(w_bas[s_idxs[s], i] * n_available_bins), BIN_MAX)
                    if b not in w_ap_final.keys():
                        w_ap_final[b] = [result[K_W_AP][-1, i]]
                    else:
                        w_ap_final[b] += [result[K_W_AP][-1, i]]

            for b in range(len(w_ap_final)):
                w_ap_final_mean[s, t, b] = float(np.mean(w_ap_final[b]))

    y_ticks = [i / 5 for i in range(6)]
    y_tick_labels: list = ['' for _ in range(6)]
    y_tick_labels[0] = 0
    y_tick_labels[-1] = 1
    c_tick_labels: list = [None for _ in range(n_bins+1)]
    c_tick_labels[0] = 0
    c_tick_labels[-1] = n_bins / n_available_bins
    colors = [COL_T1, COL_T2]

    for k in range(ns):
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [COL_DISTRACTOR, colors[k]])
        for b in range(n_bins):
            axs[k].plot(theta_0, w_ap_final_mean[k, :, b], marker='.', color=cmap(b/(n_bins-1.)), ms=MARKER_SIZE)
        if diverged:  # The smallest values of theta_0 (in the fast-learning regime)
            axs[k].axvspan(theta_0[0], max(diverged) + 0.25, color=COL_DIVERGED, linewidth=0, zorder=0)
            axs[k].text((theta_0[0] + max(diverged) + 0.25) / 2, 0.5, 'Diverges', rotation=90, ha='center',
                        va='center', color='0.4')
        for t0 in partly_diverged:
            axs[k].axvspan(t0 - 0.25, t0 + 0.25, color=COL_PARTLY_DIVERGED, linewidth=0, zorder=0)
        axs[k].vlines(x=HYPERPARAMETERS['fast' if fast else 'biological']['theta_0'], ymin=0, ymax=1, color='k',
                      linestyle='--', linewidth=AXIS_WIDTH, zorder=1)
        axs[k].spines[['right', 'top']].set_visible(False)
        axs[k].spines['left'].set_bounds(0, 1)
        axs[k].spines['bottom'].set_bounds(theta_0[0], theta_0[-1])
        axs[k].set_xticks([t for t in theta_0 if t == round(t)])
        axs[k].set_yticks(y_ticks, y_tick_labels)
        axs[k].set_xlabel(r'$\theta_0$')

    for k in range(1, ns):
        axs[k].set_yticklabels([])
    axs[0].set_ylabel('$w^{ap}$')

    adjust_figure(fig=fig, w_space=0.15)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast) if plot_dir is None else plot_dir, plot_name=plot_name)


def fs12b_theta0(saving: bool = False, fast: bool = False) -> None:
    """
    Sensory dendrite synapse strength after 1800 trials depending on the bottom-up selectivity and on theta_0, with the
    value of theta_0 of the learning regime marked by a dashed line
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param fast: whether to plot the theta_0 sweep of the fast-learning regime
    """
    plot_theta0_w_ap(saving=saving, plot_name='fig_s12b_theta_0', fast=fast)


def plot_selectivity_traces(saving: bool, plot_name: str, simulation: Simulation = Simulation.DEFAULT,
                            fast: bool = False) -> None:
    """
    Evolution of the sensory pyramidal neuron selectivity for the go or no-go stimulus as learning progresses and
    depending on the bottom-up selectivity of the neurons (encoded as the difference in the initial basal synaptic
    weights between texture 1 and texture 2), as in fig. S12c, without apical inhibition.
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param simulation: Simulation.DEFAULT for the model without bottom-up plasticity or a bottom-up plasticity
    simulation
    :param fast: whether to plot the simulation of the fast-learning regime
    """

    bin_factor = 20
    set_style()
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    bin_max = 8
    n_bins = 2 * bin_max + 1
    results = get_results(simulation=simulation, fast=fast)
    x_s1 = nan_array((N_SEEDS, N_TRIALS, N_Z))
    x_s2 = nan_array((N_SEEDS, N_TRIALS, N_Z))
    bin_idxs = {}
    for s in range(N_SEEDS):
        x_s1[s, :, :] = get_texture_specific(texture=True, outcomes=results[s][K_OUTCOME], values=results[s][K_X_SOM_TXT])
        x_s2[s, :, :] = get_texture_specific(texture=False, outcomes=results[s][K_OUTCOME], values=results[s][K_X_SOM_TXT])
        w_bas = results[s][K_W_BAS]
        for i in range(N_Z):
            b = max(min(round((w_bas[T1_IDX, i] - w_bas[T2_IDX, i]).item() * bin_factor), bin_max), -bin_max)
            if b not in bin_idxs.keys():
                bin_idxs[b] = [[s, i]]
            else:
                bin_idxs[b] += [[s, i]]
    batch_size = 20
    trials = np.arange(N_TRIALS / batch_size)
    for b in bin_idxs.keys():
        s1_stack = np.vstack([x_s1[idx[0], :, idx[1]] for idx in bin_idxs[b]])
        s2_stack = np.vstack([x_s2[idx[0], :, idx[1]] for idx in bin_idxs[b]])
        if batch_size == 1:
            s1_responses = np.nanmean(s1_stack, axis=0)
            s2_responses = np.nanmean(s2_stack, axis=0)
        else:
            s1_responses = batch_nan_stat(s1_stack, batch_size=batch_size)
            s2_responses = batch_nan_stat(s2_stack, batch_size=batch_size)
        axs.plot(trials, s1_responses-s2_responses, color=GO_NOGO_MAP(b/(2. * bin_max) + 0.5))
    x_ticks = [500, 1000, 1500]
    axs.set_xticks([x/batch_size for x in x_ticks], x_ticks)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=GO_NOGO_MAP, norm=norm)
    boundaries = [b / (n_bins-1) for b in range(n_bins)]
    cbar = plt.colorbar(mappable=sm, ax=axs, boundaries=boundaries, ticks=boundaries)
    cbar.ax.set_yticklabels([f'{b / bin_factor}' if b % 4 == 0 else '' for b in range(-bin_max, bin_max+1)])
    cbar.ax.yaxis.set_tick_params(pad=1)
    # The neurons are binned by their initial basal weights, which bottom-up plasticity changes with learning
    t_0 = '(t_0)' if simulation in BU_SIMULATIONS else ''
    cbar.ax.set_ylabel(f'$w^{{bas}}_{{Go}}{t_0}-w^{{bas}}_{{NoGo}}{t_0}$', rotation=270, labelpad=10)
    p_cmap = get_performance_cm()
    cb_width = 0.04
    c_map_ax = axs.inset_axes([0., 0, 1, cb_width], zorder=-10)
    cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='horizontal')
    cb.outline.set_visible(False)
    cb.set_ticks([])
    axs.set_ylabel('$x^{som}(Go)-x^{som}(NoGo)}$')
    axs.set_xlabel('Trial')
    axs.set_xlim([0, N_TRIALS / batch_size])
    axs.spines[['right', 'top']].set_visible(False)
    axs.set_ylim([-1.15 * MAX_GAIN, 1.05 * MAX_GAIN])  # Selectivities reach +-MAX_GAIN once the apical gain is maximal
    axs.set_yticks([-MAX_GAIN, -MAX_GAIN / 2, 0, MAX_GAIN / 2, MAX_GAIN])
    axs.spines['left'].set_bounds(-MAX_GAIN, MAX_GAIN)

    adjust_figure(fig=fig, h_space=0.1)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def fs12c_selectivity_traces(saving: bool = False, fast: bool = False) -> None:
    """
    Evolution of the sensory pyramidal neuron selectivity for the go or no-go stimulus as learning progresses and
    depending on the bottom-up selectivity of the neurons (encoded as the difference in the basal synaptic weights
    between texture 1 and texture 2).
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    plot_selectivity_traces(saving=saving, plot_name='fig_s12c_selectivity_traces', fast=fast)


def plot_selectivity_distributions(saving: bool, plot_name: str, simulation: Simulation = Simulation.DEFAULT,
                                   phases: tuple[tuple[int, int], ...] = ((0, N_PHASE_TRIALS),
                                                                          (N_TRIALS - N_PHASE_TRIALS, N_TRIALS)),
                                   width: float = None, y_max: float = 0.15, legend_loc: str = 'lower right',
                                   legend_anchor: tuple[float, float] = (1.03, 0.08), fast: bool = False) -> None:
    """
    Distribution (over all neurons that had a minimal response in the texture window) of the go-stimulus vs. the
    no-go stimulus selectivity in ranges of trials (by default before and after learning), as in fig. S12d, without
    apical inhibition. Non-selective neurons are shown as neurons that had a minimal response in the texture window,
    but were not selective for either stimulus.
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param simulation: Simulation.DEFAULT for the model without bottom-up plasticity or a bottom-up plasticity
    simulation
    :param phases: first trial and last trial plus one of each range of trials, one histogram per range
    :param width: width of the figure (by default PANEL_WIDTH per histogram)
    :param y_max: upper limit of the histograms, as a fraction of all neurons (ticks every 5 percent)
    :param legend_loc: location of the legend in the last histogram
    :param legend_anchor: anchor of the legend in the axes coordinates of the last histogram
    :param fast: whether to plot the simulation of the fast-learning regime
    """

    set_style()
    results = get_results(simulation=simulation, fast=fast)
    nn = N_SEEDS * N_Z
    phase = list(phases)
    n_phases = len(phase)
    fig, axs = plt.subplots(nrows=1, ncols=n_phases, figsize=(width or n_phases*PANEL_WIDTH, PANEL_HEIGHT),
                            squeeze=False)
    axs = axs[0]
    color_unresponsive = 'white'
    color_unselective = COL_DISTRACTOR
    for i in range(n_phases):
        responsive = [get_texture_responsive(result=r, trials=phase[i]) for r in results]
        selectivities = np.concatenate([get_texture_selectivity(result=r, trials=phase[i])[resp]
                                        for r, resp in zip(results, responsive)])
        counts = plot_selectivity_histogram(ax=axs[i], selectivities=selectivities)
        ax_ins = axs[i].inset_axes([0.0, 0.5, 0.5, 0.5])
        numbers = [nn - len(selectivities)] + list(counts)
        n_tot = sum(numbers)
        labels = [f'{100 * n / n_tot:.1f}%' for n in numbers]
        ax_ins.pie(numbers, labels=labels, labeldistance=1.2,
                   colors=[color_unresponsive, color_unselective, COL_T1, COL_T2],
                   wedgeprops={'linewidth': AXIS_WIDTH, 'edgecolor': 'k'})
    for ax in axs[1:]:
        ax.set_yticklabels([])
    ticks_y = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(100 * x / nn))
    axs[0].set_ylabel('Neurons [%]')
    for i in range(n_phases):
        axs[i].yaxis.set_major_formatter(ticks_y)
        axs[i].set_xlabel('$x^{som}(go)-x^{som}(no-go)}$')
        axs[i].spines[['right', 'top']].set_visible(False)
        axs[i].set_ylim([0, y_max * nn])
        axs[i].set_yticks([0.05 * i * nn for i in range(int(round(y_max / 0.05)) + 1)])

    p1 = matplotlib.patches.Patch(color=COL_T1, label='go selective')
    p2 = matplotlib.patches.Patch(color=COL_T2, label='no-go selective')
    p3 = matplotlib.patches.Patch(color=color_unselective, label='non-selective')
    p4 = matplotlib.patches.Patch(color=color_unresponsive, label='unresponsive', lw=AXIS_WIDTH, ec='k')
    axs[-1].legend(handles=[p1, p2, p3, p4], handlelength=1, loc=legend_loc, bbox_to_anchor=legend_anchor,
                   borderaxespad=0., frameon=False)

    adjust_figure(fig=fig, h_space=0.1)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def fs12cd_selectivity_distribution(saving: bool = False, fast: bool = False) -> None:
    """
    Distribution (over all neurons that had a minimal response in the texture window) of the go-stimulus vs. the
    no-go stimulus selectivity before and after learning. Non-selective neurons are shown as neurons that had a minimal
    response in the texture window, but were not selective for either stimulus.
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    plot_selectivity_distributions(saving=saving, plot_name='fig_s12c_selectivity_distributions', fast=fast)


def fs13_apical_dendrites_traces_boxplots(saving: bool = True, verbose: bool = False, fast: bool = False) -> None:
    """
    Evolution of apical activity with learning during different trial windows and for each dendrite type and each trial
    type (Hit, CR, Miss, FA).
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param verbose: whether to print out p-values
    :param fast: whether to plot the simulation of the fast-learning regime
    """
    for outcome in (Outcome.HIT, Outcome.CR, Outcome.FA):
        plot_apical_trace_boxplot(saving=saving, verbose=verbose, outcome=outcome, fast=fast)


def plot_bu_performance(saving: bool, plot_name: str, inhibition: bool = False, fast: bool = False) -> None:
    """
    Mean performance traces (+- standard deviation over seeds) of the control without bottom-up plasticity and of the
    bottom-up plasticity rules, in the protocol with or without apical inhibition
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param inhibition: whether to plot the protocol with the apical dendrites inhibited for the first N_TRIALS trials
    :param fast: whether to plot the simulations of the fast-learning regime
    """
    set_style()
    conditions = get_bu_conditions(fast=fast)
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    n_trials = get_n_trials(simulation=Simulation.APICAL_INHIBITION if inhibition else Simulation.DEFAULT)
    ax.plot([0, n_trials], [0.5, 0.5], 'k:')
    for simulation in conditions:
        results = get_bu_results(simulation=simulation, inhibition=inhibition, fast=fast)
        perfs = np.array([r[K_PERFORMANCE] for r in results])
        trials = np.arange(perfs.shape[1])
        y = {'mean': np.mean(perfs, axis=0), 'std': np.std(perfs, axis=0)}
        plot_trace(trials, y, color=BU_COLORS[simulation], alpha=0.15, label=BU_LABELS[simulation], ax=ax)
    y_min = 0.25
    plot_inhibition_bar(ax=ax, y=0.03, n_trials=n_trials, inhibited=inhibition)
    ax.text(-0.02, 0., "Apical\ninhibition", transform=ax.transAxes, ha="right")
    ax.set_xlim([0, n_trials])
    ax.set_ylim([y_min, 1])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=["", "", "0.6", "", "0.8", "", "1.0"])
    if inhibition:
        ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500], labels=[0, "", 1000, "", 2000, "", 3000, ""])
        ax.legend(loc='upper left', frameon=False, handlelength=1)  # Every condition is at chance in this region
    else:
        ax.set_xticks([0, 500, 1000, 1500])
    ax.set_ylabel("Performance")
    ax.set_xlabel("Trial")
    ax.yaxis.set_label_coords(-0.18, 0.6)
    ax.spines[['top', 'right']].set_visible(False)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def plot_bu_expert_trials(saving: bool, plot_name: str, fast: bool = False, verbose: bool = False,
                          x_max: int = N_TRIALS) -> None:
    """
    Number of trials to reach expert performance of the control without bottom-up plasticity and of the bottom-up
    plasticity rules, without apical inhibition (unperturbed) and after the end of the apical inhibition (corrected:
    counted from trial N_TRIALS on), with the significance of the difference to the control in the same protocol and
    of the difference between the two protocols
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param fast: whether to plot the simulations of the fast-learning regime
    :param verbose: boolean encoding whether to print the expert trials and the statistical tests
    :param x_max: upper limit of the trial axis
    """
    set_style()
    conditions = get_bu_conditions(fast=fast)
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    offsets = (0.19, -0.19)  # The unperturbed bar above the corrected one
    xp_ts = {}
    for i, simulation in enumerate(conditions):
        color = BU_COLORS[simulation]
        for j, inhibition in enumerate((False, True)):
            results = get_bu_results(simulation=simulation, inhibition=inhibition, fast=fast)
            xp_t = np.array([r[K_EXPERT_T] for r in results], dtype=float) - (N_TRIALS if inhibition else 0)
            xp_ts[simulation, inhibition] = xp_t
            y = -i + offsets[j]
            face = 'white' if inhibition else matplotlib.colors.to_rgba(color, 0.35)
            ax.barh(y, np.nanmean(xp_t), height=0.3, linewidth=1., edgecolor=color, color=face, align='center')
            ax.errorbar(np.nanmean(xp_t), y, xerr=np.nanstd(xp_t), color='k', capsize=2., linewidth=AXIS_WIDTH,
                        capthick=AXIS_WIDTH)
            ax.plot(xp_t, [y] * len(xp_t), 'ko', markersize=1)
            if verbose:
                print(f'{BU_LABELS[simulation]} {"corrected" if inhibition else "unperturbed"}: '
                      f'{np.nanmean(xp_t):.0f} ± {np.nanstd(xp_t):.0f} (not expert: {np.sum(np.isnan(xp_t))})')

    # Significance of the difference to the control in the same protocol (next to the bars) and between the protocols
    # (brackets on the right)
    x_bracket = 0.99 * x_max
    for i, simulation in enumerate(conditions):
        for j, inhibition in enumerate((False, True)):
            if simulation != Simulation.DEFAULT:
                p_value = stats.ranksums(xp_ts[simulation, inhibition], xp_ts[Simulation.DEFAULT, inhibition])[1]
                s = get_significance(p_value)
                x = np.nanmax(xp_ts[simulation, inhibition]) + 0.03 * x_max
                ax.text(x, -i + offsets[j], s, va='center', ha='left', weight='normal' if s == 'n.s.' else 'bold')
                if verbose:
                    print(f'{BU_LABELS[simulation]} vs. control {"corrected" if inhibition else "unperturbed"}: {s} '
                          f'({p_value:.1e})')
        p_value = stats.ranksums(xp_ts[simulation, False], xp_ts[simulation, True])[1]
        s = get_significance(p_value)
        ax.plot([x_bracket, x_bracket], [-i + offsets[1], -i + offsets[0]], 'k-', linewidth=AXIS_WIDTH)
        ax.text(x_bracket + 0.02 * x_max, -i, s, rotation=270, va='center', ha='left',
                weight='normal' if s == 'n.s.' else 'bold')
        if verbose:
            print(f'{BU_LABELS[simulation]} unperturbed vs. corrected: {s} ({p_value:.1e})')

    handles = [matplotlib.patches.Patch(facecolor=(0.5, 0.5, 0.5, 0.35), edgecolor='grey', label='Unperturbed'),
               matplotlib.patches.Patch(facecolor='white', edgecolor='grey', label='Corrected')]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(0.42, 0.45), frameon=False, handlelength=1)
    ax.set_yticks([-i for i in range(len(conditions))], [BU_LABELS[s] for s in conditions])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim([-len(conditions) + 0.5, 0.5])
    ax.set_xlim([0, x_max])
    ax.set_xticks(np.arange(0, x_max, 500 if x_max > 1000 else 100))
    ax.set_xlabel("Trials to expert")
    ax.spines[['top', 'right', 'left']].set_visible(False)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def plot_bu_basal_weights(saving: bool, plot_name: str, fast: bool = False, n_ranks: int = 8) -> None:
    """
    Profile of the basal weights of the neurons at the end of the simulation without apical inhibition: the basal
    weights of each neuron (from the tone, the two textures and the distractors) are sorted from the largest (rank 1) to
    the smallest, and the weight at each rank is averaged over the neurons of all seeds, for the control without
    bottom-up plasticity (whose weights are the initial weights of all the simulations) and for the bottom-up plasticity
    rules
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param fast: whether to plot the simulations of the fast-learning regime
    :param n_ranks: number of the strongest inputs of each neuron to plot
    """
    set_style()
    conditions = get_bu_conditions(fast=fast)
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    ranks = np.arange(1, n_ranks + 1)
    for simulation in conditions:
        results = get_bu_results(simulation=simulation, fast=fast)
        w_bas = np.stack([r[K_W_BAS_FINAL] for r in results])  # Seeds x basal inputs x neurons
        profile = np.mean(-np.sort(-w_bas, axis=1)[:, :n_ranks], axis=(0, 2))
        label = 'Control (initial)' if simulation == Simulation.DEFAULT else BU_LABELS[simulation]
        ax.plot(ranks, profile, color=BU_COLORS[simulation], marker='o', markersize=MARKER_SIZE, label=label)
    ax.set_xticks(ranks)
    ax.set_xlim([0.5, n_ranks + 0.5])
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('Rank of the basal input in the neuron\n(1 = largest weight)')
    ax.set_ylabel('Mean basal weight $w^\\mathrm{bas}$')
    ax.legend(loc='upper right', frameon=False, handlelength=1)
    ax.spines[['top', 'right']].set_visible(False)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def plot_bu_detectors(saving: bool, plot_name: str, fast: bool = False) -> None:
    """
    Evolution of the percentage of texture detectors (neurons with a basal weight above DETECTOR_THRESHOLD from either
    texture, see get_texture_detectors) among all neurons, from the snapshots of the basal weights every SNAPSHOT_EVERY
    trials, of the control without bottom-up plasticity and of the bottom-up plasticity rules, without apical inhibition
    (solid lines) and in the apical inhibition protocol (dashed lines), mean over seeds
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param fast: whether to plot the simulations of the fast-learning regime
    """
    set_style()
    conditions = get_bu_conditions(fast=fast)
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    y_max = 0.
    for simulation in conditions:
        for inhibition in (True, False):
            results = get_bu_results(simulation=simulation, inhibition=inhibition, fast=fast)
            detectors = []
            trials = np.zeros(0)
            for r in results:
                trials, w_bas = get_basal_history(result=r)
                detectors.append([100 * np.mean(get_texture_detectors(w_bas=w)) for w in w_bas])
            detectors = np.mean(detectors, axis=0)
            y_max = max(y_max, np.max(detectors))
            ax.plot(trials, detectors, color=BU_COLORS[simulation], linestyle='--' if inhibition else '-')
    for k, inhibition in enumerate((False, True)):
        plot_inhibition_bar(ax=ax, y=0.08 - 0.05 * k, n_trials=get_n_trials(
            simulation=Simulation.APICAL_INHIBITION if inhibition else Simulation.DEFAULT), inhibited=inhibition,
                            height=0.025)
    ax.text(-0.02, 0., "Apical\ninhibition", transform=ax.transAxes, ha="right")
    y_lim = 1.15 * y_max
    ax.set_ylim([-0.25 * y_lim, y_lim])  # The bars below the traces as in fig. 4h
    ax.set_yticks(np.arange(0, y_lim, 5))
    ax.spines['left'].set_bounds(0, y_lim)
    handles = [matplotlib.lines.Line2D([], [], color='k', linestyle=ls, label=label)
               for ls, label in (('-', 'Unperturbed'), ('--', 'Inhibition protocol'))]
    ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1., 0.15), frameon=False, handlelength=1.5)
    ax.set_xlim([0, N_TRIALS_AP_INH])
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500], labels=[0, "", 1000, "", 2000, "", 3000, ""])
    ax.set_xlabel('Trial')
    ax.set_ylabel('Texture detectors\n[% of neurons]')
    ax.yaxis.set_label_coords(-0.14, 0.6)
    ax.spines[['top', 'right']].set_visible(False)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def plot_bu_selectivity_classes(saving: bool, plot_name: str, fast: bool = False) -> None:
    """
    Fractions of go-selective, no-go-selective, non-selective and unresponsive neurons (as in fig. S12d) in the first and
    in the last N_PHASE_TRIALS trials without apical inhibition, for the control without bottom-up plasticity and for the
    bottom-up plasticity rules
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param fast: whether to plot the simulations of the fast-learning regime
    """
    set_style()
    conditions = get_bu_conditions(fast=fast)
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    phases = ((0, N_PHASE_TRIALS), (N_TRIALS - N_PHASE_TRIALS, N_TRIALS))
    phase_labels = (f'First {N_PHASE_TRIALS} trials', f'Last {N_PHASE_TRIALS} trials')
    offsets = (0.19, -0.19)
    colors = (COL_T1, COL_T2, COL_DISTRACTOR, 'white')
    for i, simulation in enumerate(conditions):
        results = get_bu_results(simulation=simulation, fast=fast)
        for j, phase in enumerate(phases):
            fractions = 100 * np.mean([get_selectivity_classes(result=r, trials=phase) for r in results], axis=0)
            y = -i + offsets[j]
            ax.barh(y, fractions, left=np.cumsum(fractions) - fractions, height=0.3, color=colors, linewidth=0)
            ax.barh(y, 100, height=0.3, facecolor='none', edgecolor='k', linewidth=AXIS_WIDTH)
            if i == 0:
                ax.text(98, y, phase_labels[j], va='center', ha='right', fontsize=FONT_SIZE - 1)
    handles = [matplotlib.patches.Patch(color=COL_T1, label='go selective'),
               matplotlib.patches.Patch(color=COL_T2, label='no-go selective'),
               matplotlib.patches.Patch(color=COL_DISTRACTOR, label='non-selective'),
               matplotlib.patches.Patch(color='white', label='unresponsive', lw=AXIS_WIDTH, ec='k')]
    ax.legend(handles=handles, handlelength=1, loc='lower left', bbox_to_anchor=(-0.32, 1.), ncol=2, borderaxespad=0.,
              frameon=False, columnspacing=1.)
    ax.set_yticks([-i for i in range(len(conditions))], [BU_LABELS[s] for s in conditions])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim([-len(conditions) + 0.5, 0.5])
    ax.set_xlim([0, 100])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Neurons [%]')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def plot_bu_selectivity_distributions(saving: bool, plot_name: str, fast: bool = False) -> None:
    """
    Distributions of the texture selectivity of the texture-responsive neurons in the last N_PHASE_TRIALS trials
    without apical inhibition (as in fig. S12d), for the control without bottom-up plasticity and for the bottom-up
    plasticity rules
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param fast: whether to plot the simulations of the fast-learning regime
    """
    set_style()
    conditions = get_bu_conditions(fast=fast)
    n_conditions = len(conditions)
    fig, axs = plt.subplots(nrows=1, ncols=n_conditions, figsize=(2 * PANEL_WIDTH, PANEL_HEIGHT), sharey='all')
    phase = (N_TRIALS - N_PHASE_TRIALS, N_TRIALS)
    nn = N_SEEDS * N_Z
    for ax, simulation in zip(axs, conditions):
        results = get_bu_results(simulation=simulation, fast=fast)
        selectivities = np.concatenate([get_texture_selectivity(result=r, trials=phase)[
                                            get_texture_responsive(result=r, trials=phase)] for r in results])
        plot_selectivity_histogram(ax=ax, selectivities=selectivities)
        ax.set_title(BU_LABELS[simulation])
        ax.set_xticks([-MAX_GAIN, 0, MAX_GAIN])
        ax.spines[['right', 'top']].set_visible(False)
    axs[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(100 * x / nn)))
    axs[0].set_ylim([0, 0.15 * nn])
    axs[0].set_yticks([0.05 * i * nn for i in range(4)])
    axs[0].set_ylabel('Neurons [%]')
    fig.supxlabel('$x^{som}(go)-x^{som}(no-go)}$', fontsize=FONT_SIZE, y=0.)
    adjust_figure(fig=fig, w_space=0.3)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def plot_bu_apical_responses(saving: bool, plot_name: str, fast: bool = False) -> None:
    """
    Mean apical activity (+- standard deviation over seeds) of the outcome dendrites at the outcome and of the sensory
    dendrites at the touch of the texture-responsive neurons (as in fig. 4g) before the learning trial (naive) and from
    the expert trial on (expert), without apical inhibition, for the control without bottom-up plasticity and for the
    bottom-up plasticity rules
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param fast: whether to plot the simulations of the fast-learning regime
    """
    set_style()
    conditions = get_bu_conditions(fast=fast)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(PANEL_WIDTH, PANEL_HEIGHT), sharey='all')
    titles = ('Outcome\ndendrites', 'Sensory\ndendrites')
    window_labels = ('outcome', 'touch')
    for i, simulation in enumerate(conditions):
        color = BU_COLORS[simulation]
        results = get_bu_results(simulation=simulation, fast=fast)
        activity = nan_array((2, 2, len(results)))  # Dendrite type x learning phase (naive, expert) x seed
        for s, r in enumerate(results):
            sensory = get_sensory_dendrites(result=r, neurons=get_texture_responsive(result=r))
            dendrites = (r[K_DENDRITE_OUTCOME], sensory[:, TEXTURE_T - TONE_T])
            if np.isfinite(r[K_LEARNING_T]) and np.isfinite(r[K_EXPERT_T]):
                for k in range(2):
                    activity[k, 0, s] = np.nanmean(dendrites[k][:int(r[K_LEARNING_T])])
                    activity[k, 1, s] = np.nanmean(dendrites[k][int(r[K_EXPERT_T]):])
        for k in range(2):
            means, stds = np.nanmean(activity[k], axis=1), np.nanstd(activity[k], axis=1)
            axs[k].plot(means, [-i, -i], color=color, zorder=1)
            for p, face in enumerate(('white', color)):
                axs[k].errorbar(means[p], -i, xerr=stds[p], marker='o', markersize=2 * MARKER_SIZE,
                                markerfacecolor=face, markeredgecolor=color, ecolor=color, linewidth=AXIS_WIDTH,
                                markeredgewidth=AXIS_WIDTH * 1.5, capsize=0, zorder=2)
    for k in range(2):
        axs[k].set_title(titles[k])
        axs[k].set_xlim([0, 1])
        axs[k].set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
        axs[k].set_xlabel(f'Activity at {window_labels[k]}')
        axs[k].spines[['top', 'right', 'left']].set_visible(False)
        axs[k].tick_params(axis='y', length=0)
    axs[0].set_yticks([-i for i in range(len(conditions))], [BU_LABELS[s] for s in conditions])
    axs[0].set_ylim([-len(conditions) + 0.5, 0.5])
    handles = [matplotlib.lines.Line2D([], [], marker='o', markersize=2 * MARKER_SIZE, markerfacecolor=face,
                                       markeredgecolor='k', linestyle='none', label=label)
               for face, label in (('white', 'Naïve'), ('k', 'Expert'))]
    axs[1].legend(handles=handles, loc='center right', bbox_to_anchor=(1., 0.36), frameon=False, handletextpad=0.1,
                  borderaxespad=0.)
    adjust_figure(fig=fig, w_space=0.15)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def plot_paired_expert_trials(saving: bool, plot_name: str, plot_dir: Path, expert_trials: dict,
                              labels: tuple[str, str], colors: tuple[str, str], verbose: bool = False) -> None:
    """
    Trials to expert performance of each seed in two conditions, without apical inhibition (unperturbed) and after the
    end of the apical inhibition (corrected: counted from trial N_TRIALS on). The seeds are paired (a seed draws the
    same initial weights and stimuli in both conditions; lines), and the significance is that of the paired difference
    (Wilcoxon signed-rank test)
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param plot_dir: directory in which to save the plot
    :param expert_trials: for each protocol (False: unperturbed, True: corrected) the expert trials of each seed in the
    two conditions (conditions x seeds, nan if not expert)
    :param labels: labels of the two conditions (a space breaks the tick label into two lines)
    :param colors: colors of the markers of the two conditions
    :param verbose: boolean encoding whether to print the expert trials and the statistical tests
    """
    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    protocol_labels = ('Unperturbed', 'Corrected')
    group_offset = 2.6  # Distance between the two protocols on the x-axis
    y_max = N_TRIALS
    n_lines = max(label.count(' ') + 1 for label in labels)  # Lines of the tick labels
    for j, inhibition in enumerate((False, True)):
        xp_ts = expert_trials[inhibition]
        x = np.array([0., 1.]) + group_offset * j
        ax.plot(x, xp_ts, color='grey', linewidth=AXIS_WIDTH, alpha=0.6, zorder=1)
        for k in range(2):
            ax.plot([x[k]] * xp_ts.shape[1], xp_ts[k], 'o', color=colors[k], markersize=MARKER_SIZE, zorder=2)
            ax.plot(x[k] + np.array([-0.22, 0.22]), [np.nanmean(xp_ts[k])] * 2, 'k-', linewidth=2 * LINE_WIDTH,
                    zorder=3)
        paired = np.all(np.isfinite(xp_ts), axis=0)  # Seeds that become expert in both conditions
        difference = xp_ts[0, paired] - xp_ts[1, paired]
        p_value = stats.wilcoxon(difference)[1]
        s = get_significance(p_value)
        ax.plot(x, [1.03 * y_max] * 2, 'k-', linewidth=AXIS_WIDTH)
        ax.text(np.mean(x), 1.03 * y_max, s, ha='center', va='bottom', weight='normal' if s == 'n.s.' else 'bold')
        # Under the tick labels, at a distance from the axis in points (which the layout keeps, unlike a fraction of
        # the axes height): 22 points under two lines
        ax.annotate(protocol_labels[j], xy=(np.mean(x), 0.), xycoords=ax.get_xaxis_transform(),
                    xytext=(0., -8. - 7. * n_lines), textcoords='offset points', ha='center', va='top')
        if verbose:
            print(f'{protocol_labels[j]}: {labels[0]} {np.nanmean(xp_ts[0]):.0f}, {labels[1]} '
                  f'{np.nanmean(xp_ts[1]):.0f}, faster in {np.sum(difference > 0)} of {len(difference)} seeds by '
                  f'{np.mean(difference):.0f} trials (not expert: {np.sum(~paired)}), signed-rank {s} ({p_value:.1e})')
    # The labels on two lines at their spaces (one line of the rule's label would overlap the control's)
    ax.set_xticks([0, 1, group_offset, group_offset + 1], [label.replace(' ', '\n') for label in labels] * 2)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim([-0.5, group_offset + 1.5])
    ax.set_ylim([0, 1.12 * y_max])
    ax.set_yticks([0, 500, 1000, 1500])
    ax.spines['left'].set_bounds(0, y_max)
    ax.set_ylabel('Trials to expert')
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=plot_dir, plot_name=plot_name)


def plot_bu_paired_expert_trials(saving: bool, plot_name: str, simulation: Simulation = Simulation.BU_APICAL,
                                 fast: bool = False, verbose: bool = False) -> None:
    """
    Trials to expert performance of each seed in the control without bottom-up plasticity and with a bottom-up
    plasticity rule, paired by seed, without apical inhibition and after its end (see plot_paired_expert_trials)
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param simulation: the bottom-up plasticity simulation to compare with the control
    :param fast: whether to plot the simulations of the fast-learning regime
    :param verbose: boolean encoding whether to print the expert trials and the statistical tests
    """
    conditions = (Simulation.DEFAULT, simulation)
    expert_trials = {inhibition: np.array(
        [[r[K_EXPERT_T] for r in get_bu_results(simulation=s, inhibition=inhibition, fast=fast)] for s in conditions],
        dtype=float) - (N_TRIALS if inhibition else 0) for inhibition in (False, True)}  # Conditions x seeds
    plot_paired_expert_trials(saving=saving, plot_name=plot_name, plot_dir=get_plot_dir(fast),
                              expert_trials=expert_trials, labels=tuple(BU_LABELS[s] for s in conditions),
                              colors=tuple(BU_COLORS[s] for s in conditions), verbose=verbose)


def plot_bu_change_by_apical_weight(saving: bool, plot_name: str, fast: bool = False) -> None:
    """
    Change of the basal weights of each neuron from their initial values to the end of the simulation without apical
    inhibition (see get_basal_change) for the bottom-up plasticity rules: mean +- standard deviation over seeds of the
    mean over the neurons recruited by the apical loop (final apical weight above RECRUITED_THRESHOLD, filled) and over
    the other neurons (open)
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param fast: whether to plot the simulations of the fast-learning regime
    """
    set_style()
    simulations = get_bu_simulations(fast=fast)
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    for i, simulation in enumerate(simulations):
        color = BU_COLORS[simulation]
        change = []  # Seeds x (other neurons, recruited neurons)
        for r in get_bu_results(simulation=simulation, fast=fast):
            recruited = get_recruited(result=r)
            final_change = get_basal_change(result=r)[1][-1]
            change.append([np.mean(final_change[~recruited]), np.mean(final_change[recruited])])
        means, stds = np.mean(change, axis=0), np.std(change, axis=0)
        ax.plot(means, [-i, -i], color=color, zorder=1)
        for p, face in enumerate(('white', color)):
            ax.errorbar(means[p], -i, xerr=stds[p], marker='o', markersize=2 * MARKER_SIZE, markerfacecolor=face,
                        markeredgecolor=color, ecolor=color, linewidth=AXIS_WIDTH, markeredgewidth=AXIS_WIDTH * 1.5,
                        capsize=0, zorder=2)
    ax.set_yticks([-i for i in range(len(simulations))], [BU_LABELS[s] for s in simulations])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim([-len(simulations) + 0.5, 0.5])
    ax.set_xlim([-0.1, 3.3])  # The open markers at zero change entirely visible
    ax.set_xticks([0, 1, 2, 3])
    ax.spines['bottom'].set_bounds(0, 3)
    ax.set_xlabel('Change of the basal weights per neuron\n'
                  '$\\Sigma_k\\,|w^\\mathrm{bas}_k(1800) - w^\\mathrm{bas}_k(0)|$')
    handles = [matplotlib.lines.Line2D([], [], marker='o', markersize=2 * MARKER_SIZE, markerfacecolor=face,
                                       markeredgecolor='k', linestyle='none', label=label)
               for face, label in (('white', f'$w^\\mathrm{{ap}} \\leq {RECRUITED_THRESHOLD:g}$'),
                                   ('k', f'$w^\\mathrm{{ap}} > {RECRUITED_THRESHOLD:g}$'))]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0., 1.), ncol=2, frameon=False, handletextpad=0.1,
              borderaxespad=0., columnspacing=1.)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def plot_bu_recruited_weights(saving: bool, plot_name: str, simulation: Simulation = Simulation.BU_APICAL,
                              fast: bool = False) -> None:
    """
    Evolution of the basal weights of the neurons recruited by the apical loop (final apical weight above
    RECRUITED_THRESHOLD) with a bottom-up plasticity rule, without apical inhibition: the weight from the tone, from the
    preferred texture of each neuron (the one with the larger initial weight), from its other texture and summed over
    its distractors (see get_input_class_weights), from the snapshots every SNAPSHOT_EVERY trials, mean +- standard
    deviation over seeds of the mean over the recruited neurons
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param simulation: the bottom-up plasticity simulation
    :param fast: whether to plot the simulations of the fast-learning regime
    """
    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    weights = []
    trials = np.zeros(0)
    for r in get_bu_results(simulation=simulation, fast=fast):
        trials, w_bas = get_basal_history(result=r)
        weights.append(get_input_class_weights(w_bas=w_bas, w_bas_0=r[K_W_BAS], neurons=get_recruited(result=r)))
    weights = np.stack(weights)  # Seeds x snapshots x input classes
    labels = ('Tone', 'Preferred texture', 'Other texture', 'Distractors (sum)')
    colors = (COL_PRE, COL_TEXTURE, COL_TEXTURE, COL_DISTRACTOR)
    line_styles = ('-', '-', '--', '-')
    for c in range(4):
        y = {'mean': np.mean(weights[:, :, c], axis=0), 'std': np.std(weights[:, :, c], axis=0)}
        plot_trace(trials, y, color=colors[c], alpha=0.2, ax=ax, line_style=line_styles[c])
    ax.set_xlim([0, N_TRIALS])
    ax.set_xticks([0, 500, 1000, 1500])
    ax.set_ylim([0, 1.7])  # Room for the legend above the traces
    ax.set_yticks([0, 0.5, 1, 1.5])
    ax.spines['left'].set_bounds(0, 1.5)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Basal weight $w^\\mathrm{bas}$\nof the recruited neurons')
    ax.spines[['top', 'right']].set_visible(False)
    adjust_figure(fig=fig)
    # Legend in two rows in the margin above the axes (the traces leave no room for it inside), right-aligned with the
    # axes; the handles in column order, so that the rows read tone, preferred texture, other texture, distractors
    handles = [matplotlib.lines.Line2D([], [], color=colors[c], linestyle=line_styles[c], label=labels[c])
               for c in (0, 2, 1, 3)]
    fig.legend(handles=handles, ncol=2, loc='upper right', bbox_to_anchor=(ax.get_position().x1, 1.), frameon=False,
               handlelength=1.5, borderaxespad=0., borderpad=0., labelspacing=0.2, columnspacing=1.)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def plot_bu_weight_change(saving: bool, plot_name: str, fast: bool = False) -> None:
    """
    Evolution of the change of the basal weights from their initial values (see get_basal_change), mean over the neurons
    and the seeds, for the bottom-up plasticity rules without apical inhibition (solid lines) and in the apical
    inhibition protocol (dashed lines), from the snapshots every SNAPSHOT_EVERY trials
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param plot_name: file name under which to save the plot
    :param fast: whether to plot the simulations of the fast-learning regime
    """
    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    y_max = 0.
    for simulation in get_bu_simulations(fast=fast):
        for inhibition in (True, False):
            changes = []
            trials = np.zeros(0)
            for r in get_bu_results(simulation=simulation, inhibition=inhibition, fast=fast):
                trials, change = get_basal_change(result=r)
                changes.append(np.mean(change, axis=1))
            changes = np.mean(changes, axis=0)
            y_max = max(y_max, np.max(changes))
            ax.plot(trials, changes, color=BU_COLORS[simulation], linestyle='--' if inhibition else '-')
    for k, inhibition in enumerate((False, True)):
        plot_inhibition_bar(ax=ax, y=0.08 - 0.05 * k, n_trials=get_n_trials(
            simulation=Simulation.APICAL_INHIBITION if inhibition else Simulation.DEFAULT), inhibited=inhibition,
                            height=0.025)
    ax.text(-0.02, 0., "Apical\ninhibition", transform=ax.transAxes, ha="right")
    y_lim = 1.45 * y_max  # Room for the legend above the traces
    ax.set_ylim([-0.25 * y_lim, y_lim])  # The bars below the traces as in fig. 4h
    ax.set_yticks(np.arange(0, y_max + 0.01, 0.5))
    ax.spines['left'].set_bounds(0, y_lim)
    handles = [matplotlib.lines.Line2D([], [], color='k', linestyle=ls, label=label)
               for ls, label in (('-', 'Unperturbed'), ('--', 'Inhibition protocol'))]
    ax.legend(handles=handles, loc='upper right', frameon=False, handlelength=1.5, borderaxespad=0.)
    ax.set_xlim([0, N_TRIALS_AP_INH])
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500], labels=[0, "", 1000, "", 2000, "", 3000, ""])
    ax.set_xlabel('Trial')
    ax.set_ylabel('Change of the basal\nweights per neuron')
    ax.yaxis.set_label_coords(-0.14, 0.6)
    ax.spines[['top', 'right']].set_visible(False)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=get_plot_dir(fast), plot_name=plot_name)


def fs14b_performance(saving: bool = True) -> None:
    """
    Performance traces with the apical bottom-up plasticity rule, with and without apical inhibition (as fig. 4h)
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    plot_inhibition_performance(saving=saving, plot_name='fig_s14b_performance', simulation=Simulation.BU_APICAL)


def fs14c_paired_expert_trials(saving: bool = True, verbose: bool = False) -> None:
    """
    Trials to expert of each seed in the control and with the apical rule, without apical inhibition and after its end
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param verbose: boolean encoding whether to print the expert trials and the statistical tests
    """
    plot_bu_paired_expert_trials(saving=saving, plot_name='fig_s14c_paired_expert_trials', verbose=verbose)


def fs14d_recruited_weights(saving: bool = True) -> None:
    """
    Evolution of the basal weights of the neurons recruited by the apical loop with the apical rule
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    plot_bu_recruited_weights(saving=saving, plot_name='fig_s14d_recruited_weights')


def fs14e_selectivity_traces(saving: bool = True) -> None:
    """
    Evolution of the selectivity with learning with the apical rule (as fig. S12c)
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    plot_selectivity_traces(saving=saving, plot_name='fig_s14e_selectivity_traces', simulation=Simulation.BU_APICAL)


def fs14f_selectivity_distribution(saving: bool = True) -> None:
    """
    Distribution of the selectivity after learning with the apical rule (the after-learning part of fig. S12c/d)
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    # The axes as wide as the after-learning histogram of fig. S12c/d, which has no y-axis label; up to 20 percent, as
    # the non-selective bin at zero holds 16.9 percent of the neurons; the legend clear of the strongly selective bins
    plot_selectivity_distributions(saving=saving, plot_name='fig_s14f_selectivity_distribution',
                                   simulation=Simulation.BU_APICAL, phases=((N_TRIALS - N_PHASE_TRIALS, N_TRIALS),),
                                   width=1.12 * PANEL_WIDTH, y_max=0.2, legend_loc='upper right',
                                   legend_anchor=(1.03, 1.02))


def fs15a_performance(saving: bool = True) -> None:
    """
    Performance traces in the fast-learning regime with and without apical inhibition (as fig. 4h)
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    plot_inhibition_performance(saving=saving, plot_name='fig_s15a_performance', fast=True, plot_dir=PLOT_DIR)


def get_regime_expert_trials() -> dict[bool, np.ndarray]:
    """
    Get the trials to expert performance of each seed of the model without bottom-up plasticity in the biological and
    in the fast-learning regime. A seed draws the same initial weights and the same sequence of stimuli in both
    regimes
    :return: for each protocol (False: without apical inhibition, True: with the apical dendrites inhibited for the
    first N_TRIALS trials, counted from trial N_TRIALS on) the expert trials (biological and fast regime x seeds, nan
    if not expert)
    """
    expert_trials = {}
    for inhibition in (False, True):
        simulation = Simulation.APICAL_INHIBITION if inhibition else Simulation.DEFAULT
        expert_trials[inhibition] = np.array([[r[K_EXPERT_T] for r in get_results(simulation=simulation, fast=fast)]
                                              for fast in (False, True)], dtype=float) - (N_TRIALS if inhibition else 0)
    return expert_trials


def fs15b_paired_expert_trials(saving: bool = True, verbose: bool = False) -> None:
    """
    Trials to expert of each seed in the biological and in the fast-learning regime, without apical inhibition and after
    its end (as fig. S14c)
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param verbose: boolean encoding whether to print the expert trials and the statistical tests
    """
    plot_paired_expert_trials(saving=saving, plot_name='fig_s15b_paired_expert_trials', plot_dir=PLOT_DIR,
                              expert_trials=get_regime_expert_trials(),
                              labels=(REGIME_LABELS[False], REGIME_LABELS[True]),
                              colors=(REGIME_COLORS[False], REGIME_COLORS[True]), verbose=verbose)


def fs15c_theta0_w_ap(saving: bool = True) -> None:
    """
    Sensory dendrite synapse strength after 1800 trials depending on the bottom-up selectivity and on theta_0 in the
    fast-learning regime (as fig. S12b, with the value of theta_0 of the regime marked by a dashed line)
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    plot_theta0_w_ap(saving=saving, plot_name='fig_s15c_theta_0_w_ap', fast=True, plot_dir=PLOT_DIR)


def get_theta_0_sweep(fast: bool = False) -> tuple[list[float], np.ndarray, np.ndarray]:
    """
    Get the trials to expert and the divergence of each seed of the theta_0 sweep of the default simulation in a
    learning regime
    :param fast: whether to get the sweep of the fast-learning regime
    :return: values of theta_0, and arrays (theta_0 values x seeds) of the trials to expert (nan if not expert or
    diverged) and of whether the simulation diverged
    """
    thetas = get_thetas_0(fast=fast)
    expert, diverged = (nan_array((len(thetas), N_SEEDS)) for _ in range(2))
    for t, theta_0 in enumerate(thetas):
        for s, r in enumerate(get_results(simulation=Simulation.DEFAULT, theta_0=theta_0, fast=fast)):
            diverged[t, s] = get_diverged(result=r)
            expert[t, s] = r[K_EXPERT_T]
    return thetas, expert, diverged.astype(bool)


def plot_expert_trials_sweep(ax: plt.Axes, values: np.ndarray, expert: np.ndarray, diverged: np.ndarray,
                             value: float, edges: np.ndarray, log: bool = False) -> tuple[np.ndarray, ...]:
    """
    Plot the trials to expert performance in a parameter sweep of the fast-learning regime: mean +- standard deviation
    over the seeds that become expert, filled markers if all of them do, open markers with the number of expert seeds
    otherwise (no marker if none does), a cross at the value of the regime (on top of the markers next to it, outlined
    in white), and the band of each value at which the simulation diverges in every seed shaded (lighter if in some
    seeds only), the longest range of those labelled
    :param ax: axes to plot into
    :param values: values of the parameter in increasing order
    :param expert: trials to expert (values x seeds, nan if not expert or diverged)
    :param diverged: whether the simulation diverged (values x seeds)
    :param value: value of the parameter in the regime
    :param edges: edges of the bands of the values (one more than the values)
    :param log: whether the axis of the parameter is logarithmic (to center the label of the divergence)
    :return: number of expert seeds, mean and standard deviation of the trials to expert at each value
    """
    n_expert = np.sum(np.isfinite(expert), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means, stds = np.nanmean(expert, axis=1), np.nanstd(expert, axis=1)
    ax.plot(values, means, color=COL_FAST, zorder=1)
    for v, x in enumerate(values):
        if n_expert[v] == 0:
            continue
        own = np.isclose(x, value)
        ax.errorbar(x, means[v], yerr=stds[v], marker='none' if own else 'o', markersize=2 * MARKER_SIZE,
                    markerfacecolor=COL_FAST if n_expert[v] == N_SEEDS else 'white', markeredgecolor=COL_FAST,
                    ecolor=COL_FAST, linewidth=AXIS_WIDTH, markeredgewidth=AXIS_WIDTH * 1.5, capsize=0, zorder=2)
        if own:
            outline = matplotlib.patheffects.withStroke(linewidth=2 * AXIS_WIDTH + 0.8, foreground='white')
            ax.plot(x, means[v], marker='x', markersize=3 * MARKER_SIZE, markeredgewidth=2 * AXIS_WIDTH,
                    color=COL_FAST, linestyle='none', zorder=3, path_effects=[outline])
        if n_expert[v] < N_SEEDS:
            ax.text(x, means[v] + stds[v] + 40, f'{n_expert[v]}', ha='center', va='bottom', color=COL_FAST,
                    fontsize=FONT_SIZE - 1)
    all_diverged = np.all(diverged, axis=1)
    for v in np.flatnonzero(np.any(diverged, axis=1)):
        ax.axvspan(edges[v], edges[v + 1], color=COL_DIVERGED if all_diverged[v] else COL_PARTLY_DIVERGED, linewidth=0,
                   zorder=0)
    if all_diverged.any():
        idxs = np.flatnonzero(all_diverged)
        run = max(np.split(idxs, np.flatnonzero(np.diff(idxs) > 1) + 1), key=len)
        low, high = edges[run[0]], edges[run[-1] + 1]
        ax.text(np.sqrt(low * high) if log else (low + high) / 2, 0.5 * N_TRIALS, 'Diverges', rotation=90,
                ha='center', va='center', color='0.4')
    ax.set_ylim([0, N_TRIALS])
    ax.set_yticks([0, 500, 1000, 1500])
    ax.spines[['top', 'right']].set_visible(False)
    return n_expert, means, stds


def fs15d_theta0_expert_trials(saving: bool = True, verbose: bool = False) -> None:
    """
    Trials to expert performance in the theta_0 sweep of the fast-learning regime (see plot_expert_trials_sweep; the
    band of each value of theta_0 reaches halfway to its neighbours)
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param verbose: boolean encoding whether to print the numbers of the sweep
    """
    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    thetas, expert, diverged = get_theta_0_sweep(fast=True)
    thetas = np.array(thetas)
    edges = np.concatenate([[thetas[0]], (thetas[1:] + thetas[:-1]) / 2, [thetas[-1]]])
    n_expert, means, stds = plot_expert_trials_sweep(ax=ax, values=thetas, expert=expert, diverged=diverged,
                                                     value=HYPERPARAMETERS['fast']['theta_0'], edges=edges)
    if verbose:
        for t, theta_0 in enumerate(thetas):
            print(f'theta_0 {theta_0}: expert {n_expert[t]} of {N_SEEDS} (diverged {np.sum(diverged[t])}), '
                  f'{means[t]:.0f} +- {stds[t]:.0f}')
    ax.set_xlim([0.8, 5.2])
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel('Trials to expert')
    # Laid out with room for a title as those of fig. S15e (removed after), so that the axes of the two panels align
    ax.set_title(' ', fontsize=FONT_SIZE, pad=2)
    adjust_figure(fig=fig)
    ax.set_title('')
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name='fig_s15d_theta_0_expert_trials')


def get_learning_rate_sweep(name: str) -> tuple[list[float], np.ndarray, np.ndarray]:
    """
    Get the trials to expert and the divergence of each seed in the sweep of a learning rate of the default simulation
    in the fast-learning regime, the value of the regime included (the default simulation, read from the theta_0 sweep
    at the theta_0 of the regime, which is the same simulation)
    :param name: name of the learning rate (see helper.LEARNING_RATES)
    :return: values of the learning rate in increasing order, and arrays (values x seeds) of the trials to expert (nan
    if not expert or diverged) and of whether the simulation diverged
    """
    hp = HYPERPARAMETERS['fast']
    values = sorted(get_learning_rates(name) + [hp[name]])
    expert, diverged = (nan_array((len(values), N_SEEDS)) for _ in range(2))
    for v, value in enumerate(values):
        if value == hp[name]:
            results = get_results(simulation=Simulation.DEFAULT, theta_0=hp['theta_0'], fast=True)
        else:
            results = get_results(simulation=Simulation.DEFAULT, lr=(name, value), fast=True)
        for s, r in enumerate(results):
            diverged[v, s] = get_diverged(result=r)
            expert[v, s] = r[K_EXPERT_T]
    return values, expert, diverged.astype(bool)


def fs15e_learning_rates(saving: bool = True, verbose: bool = False) -> None:
    """
    Trials to expert performance in the sweeps of the learning rates of the fast-learning regime, each varied alone (the
    others and theta_0 at the values of the regime, see helper.get_learning_rates), against the learning rate relative
    to its value in the regime (see plot_expert_trials_sweep; the band of each value reaches the geometric midpoints
    with its neighbours)
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param verbose: boolean encoding whether to print the numbers of the sweeps
    """
    set_style()
    fig, axs = plt.subplots(nrows=1, ncols=len(LEARNING_RATES), figsize=(2 * PANEL_WIDTH, PANEL_HEIGHT), sharey=True)
    x_lim = (2 ** -3.4, 2 ** 3.4)
    w_space = 0.3  # Space between the axes, as a fraction of their width
    for ax, name in zip(axs, LEARNING_RATES):
        values, expert, diverged = get_learning_rate_sweep(name=name)
        factors = np.array(values) / HYPERPARAMETERS['fast'][name]
        edges = np.concatenate([[x_lim[0]], np.sqrt(factors[1:] * factors[:-1]), [x_lim[1]]])
        n_expert, means, stds = plot_expert_trials_sweep(ax=ax, values=factors, expert=expert, diverged=diverged,
                                                         value=1., edges=edges, log=True)
        ax.set_title(LR_LABELS[name], fontsize=FONT_SIZE, pad=2)
        ax.set_xscale('log', base=2)
        ax.set_xlim(x_lim)
        ax.set_xticks([1 / 8, 1, 8], ['1/8', '1', '8'])
        ax.set_xticks([1 / 4, 1 / 2, 2, 4], [], minor=True)
        if verbose:
            print(f'{name} ({HYPERPARAMETERS["fast"][name]:g}): ' + ', '.join(
                f'x{factor:.3g} {n_expert[v]}/{N_SEEDS} expert ({np.sum(diverged[v])} diverged) '
                f'{means[v]:.0f} +- {stds[v]:.0f}' for v, factor in enumerate(factors)))
    axs[0].set_ylabel('Trials to expert')
    # One label for the four axes, centered under them at the height of an axis label, laid out as deep as the label of
    # fig. S15d (its subscript makes it deeper; replaced after), so that the axes of the two panels align
    x_label = (len(axs) + (len(axs) - 1) * w_space) / 2
    axs[0].set_xlabel(r'$\theta_0$', x=x_label)
    adjust_figure(fig=fig, w_space=w_space)
    axs[0].set_xlabel('Learning rate (relative to the fast regime)', x=x_label)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name='fig_s15e_learning_rates')


def plot_all_fast(saving: bool = True) -> None:
    """
    Plot the panels of the model without bottom-up plasticity (figs. 4 and S11 to S13) again from the simulations of the
    fast-learning regime, under the same file names in the FAST_DIR subdirectory of PLOT_DIR (all but the transfer
    functions of fig. S11b/c, which do not depend on the learning regime)
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    f4c_performance_eg(saving=saving, fast=True)  # Performance example
    f4d_expert_v_pred(saving=saving, fast=True)  # Expert V pred trace for Hit and CR trials
    f4e_apical_dendrites_raster_plots(saving=saving, fast=True)  # S1 apical activity raster plot for Hit and CR
    f4e_soma_raster_plots(saving=saving, fast=True)  # S1 output raster plot
    f4f_sen_dendrite(saving=saving, fast=True)  # w^ap and gain
    f4g_apical_window_traces(saving=saving, fast=True)  # Apical activity evolution with learning for each time window
    f4h_performance(saving=saving, fast=True)  # Performances traces with and without apical inhibition
    f4h_expert_trials(saving=saving, fast=True)  # Expert times
    f4h_w_ap(saving=saving, fast=True)  # Top-down w^ap to S1 neurons
    fs11d_expert_v_pred(saving=saving, fast=True)  # Expert V pred trace for FA trials
    fs11e_apical_dendrites_raster_plots(saving=saving, fast=True)  # S1 apical activity raster plot for FA
    fs12a_performance_eg(saving=saving, fast=True)  # Performance example
    fs12b_wap_traces(saving=saving, fast=True)  # Evolution of apical synaptic weights
    fs12b_theta0(saving=saving, fast=True)  # Dependence on the theta_0 parameter
    fs12c_selectivity_traces(saving=saving, fast=True)  # Evolution of the selectivity with learning
    fs12cd_selectivity_distribution(saving=saving, fast=True)  # Distribution of the selectivity before and after
    fs13_apical_dendrites_traces_boxplots(saving=saving, fast=True)  # Apical activity traces and boxplots


def plot_all(saving: bool = True) -> None:
    """
    Plot all the panels, those of the fast-learning regime (see plot_all_fast) included.
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    f4c_performance_eg(saving=saving)  # Performance example for default simulation
    f4d_expert_v_pred(saving=saving)  # Expert V pred trace for Hit and CR trials
    f4e_apical_dendrites_raster_plots(saving=saving)  # S1 apical activity raster plot for Hit and CR
    f4e_soma_raster_plots(saving=saving)  # S1 output raster plot
    f4f_sen_dendrite(saving=saving)  # w^ap and gain
    f4g_apical_window_traces(saving=saving)  # Apical activity evolution with learning for different time windows
    f4h_performance(saving=saving)  # Performances traces with and without apical inhibition
    f4h_expert_trials(saving=saving)  # Expert times
    f4h_w_ap(saving=saving)  # Top-down w^ap to S1 neurons
    fs11bc_transfer_f(saving=saving)  # Transfer functions
    fs11d_expert_v_pred(saving=saving)  # Expert V pred trace for FA trials
    fs11e_apical_dendrites_raster_plots(saving=saving)  # S1 apical activity raster plot for FA
    fs12a_performance_eg(saving=saving)  # Performance example for default simulation
    fs12b_wap_traces(saving=saving)  # Evolution of apical synaptic weights
    fs12b_theta0(saving=saving)  # Dependence on the theta_0 parameter
    fs12c_selectivity_traces(saving=saving)  # Evolution of the selectivity with learning
    fs12cd_selectivity_distribution(saving=saving)  # Distribution of the selectivity before and after learning
    fs13_apical_dendrites_traces_boxplots(saving=saving)  # S1 apical activity traces and boxplots for each trial type
    fs14b_performance(saving=saving)  # Performance traces with the apical rule, with and without apical inhibition
    fs14c_paired_expert_trials(saving=saving)  # Expert trials of each seed with the apical rule and in the control
    fs14d_recruited_weights(saving=saving)  # Evolution of the basal weights of the recruited neurons
    fs14e_selectivity_traces(saving=saving)  # Evolution of the selectivity with learning with the apical rule
    fs14f_selectivity_distribution(saving=saving)  # Distribution of the selectivity after learning with the apical rule
    fs15a_performance(saving=saving)  # Performance traces in the fast-learning regime with and without inhibition
    fs15b_paired_expert_trials(saving=saving)  # Expert trials of each seed in the two regimes
    fs15c_theta0_w_ap(saving=saving)  # Dependence on the theta_0 parameter in the fast-learning regime
    fs15d_theta0_expert_trials(saving=saving)  # Expert trials in the theta_0 sweep of the fast-learning regime
    fs15e_learning_rates(saving=saving)  # Expert trials in the learning-rate sweeps of the fast-learning regime
    plot_all_fast(saving=saving)  # The panels of figs. 4 and S11 to S13 in the fast-learning regime
    print('All panels have been plotted successfully!')


if __name__ == '__main__':
    # plot_all()
    saving = False
    f4c_performance_eg(saving=saving)  # Performance example for default simulation
    f4d_expert_v_pred(saving=saving)  # Expert V pred trace for Hit and CR trials
    f4e_apical_dendrites_raster_plots(saving=saving)  # S1 apical activity raster plot for Hit and CR
    f4e_soma_raster_plots(saving=saving)  # S1 output raster plot
    f4f_sen_dendrite(saving=saving)  # w^ap and gain
    f4g_apical_window_traces(saving=saving)  # Apical activity evolution with learning for different time windows
    f4h_performance(saving=saving)  # Performances traces with and without apical inhibition
    f4h_expert_trials(saving=saving)  # Expert times
    f4h_w_ap(saving=saving)  # Top-down w^ap to S1 neurons
    fs11bc_transfer_f(saving=saving)  # Transfer functions
    fs11d_expert_v_pred(saving=saving)  # Expert V pred trace for FA trials
    fs11e_apical_dendrites_raster_plots(saving=saving)  # S1 apical activity raster plot for FA
    fs12a_performance_eg(saving=saving)  # Performance example for default simulation
    fs12b_wap_traces(saving=saving)  # Evolution of apical synaptic weights
    fs12b_theta0(saving=saving)  # Dependence on the theta_0 parameter
    fs12c_selectivity_traces(saving=saving)  # Evolution of the selectivity with learning
    fs12cd_selectivity_distribution(saving=saving)  # Distribution of the selectivity before and after learning
    fs13_apical_dendrites_traces_boxplots(saving=saving)  # S1 apical activity traces and boxplots for each trial type
    fs14b_performance(saving=saving)  # Performance traces with the apical rule, with and without apical inhibition
    fs14c_paired_expert_trials(saving=saving)  # Expert trials of each seed with the apical rule and in the control
    fs14d_recruited_weights(saving=saving)  # Evolution of the basal weights of the recruited neurons
    fs14e_selectivity_traces(saving=saving)  # Evolution of the selectivity with learning with the apical rule
    fs14f_selectivity_distribution(saving=saving)  # Distribution of the selectivity after learning with the apical rule
    fs15a_performance(saving=saving)  # Performance traces in the fast-learning regime with and without inhibition
    fs15b_paired_expert_trials(saving=saving)  # Expert trials of each seed in the two regimes
    fs15c_theta0_w_ap(saving=saving)  # Dependence on the theta_0 parameter in the fast-learning regime
    fs15d_theta0_expert_trials(saving=saving)  # Expert trials in the theta_0 sweep of the fast-learning regime
    fs15e_learning_rates(saving=saving)  # Expert trials in the learning-rate sweeps of the fast-learning regime
