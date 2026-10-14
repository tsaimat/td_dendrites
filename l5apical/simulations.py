"""
Module containing the code to simulate learning of a go/no-go sensory discrimination task by a mouse agent with and
without top-down salience signal onto apical dendrites of pyramidal neurons, with and without bottom-up (basal)
plasticity. The simulation outcomes are saved in pickle files in the 'results' directory. The performance traces are
then extracted with the method of Smith et al. (2004), either by its Python port in l5apical.smith
(compute_smith_perf()) or by the Matlab script Smith/getperfs.m, and integrated into the pickle files by
load_smith_perf().
"""
from multiprocessing import Pool
from random import seed as rd_seed
from random import shuffle as random_shuffle
from pickle import load, dump, HIGHEST_PROTOCOL
from scipy.io import savemat, loadmat
from l5apical.helper import *
from l5apical.smith import run_analysis


class Agent:
    """
    Agent class simulating how the mouse selects actions and receives rewards according to the go/no-go sensory
    discrimination task.
    """
    def __init__(self, nr_inputs: int, policy_lr: float):
        self.reward = torch.tensor([REWARD])
        self.punishment = torch.tensor([PUNISHMENT])
        self.policy_weights = torch.zeros(nr_inputs)
        self.lick_activity = torch.Tensor([0.])
        self.input_history = torch.zeros(nr_inputs)
        self.lr = policy_lr
        self.a_prob = 0.5

    def integrate_input(self, input_vec: torch.Tensor) -> None:
        """
        Append the current sensory state representation vector to the history of inputs and update the lick activity
        :param input_vec: Sensory state representation vector used as input for the policy network
        """
        self.lick_activity += torch.dot(self.policy_weights, input_vec)
        self.input_history += input_vec

    def act(self, go: int) -> tuple[torch.Tensor, bool, int]:
        """
        Sample an action, observe the resulting outcome and update the action-selection policy network
        :param go: Integer encoding the go stimulus (if non-zero) or the no-go stimulus (if zero)
        :return: Tuple containing the reward, a boolean encoding whether the lick action was performed and an integer
        encoding whether the best action was performed (1) or not (0).
        """

        # Compute the lick action probability and sample action
        lick_prob = torch.sigmoid(self.lick_activity)
        lick_prob_float = lick_prob.detach().numpy()[0]
        lick = bool(np.random.choice([0, 1], p=[1 - lick_prob_float, lick_prob_float]))

        # Get the outcome depending on the action
        if lick:
            self.a_prob = lick_prob
            if go:
                correct = 1
                reward = self.reward
            else:
                correct = 0
                reward = self.punishment
        else:
            self.a_prob = 1 - lick_prob
            reward = torch.tensor([0.])
            if go:
                correct = 0
            else:
                correct = 1

        return reward, lick, correct

    def update(self, reward: float, x_som: torch.Tensor):
        """
        Update policy network
        :param reward: The reward with which to update the weights
        :param x_som: The sensory state representation vector used as input for the policy network
        """
        self.policy_weights += self.lr * reward * (1 - self.a_prob) * self.input_history * x_som
        self.lick_activity = torch.Tensor([0.])
        self.input_history = torch.zeros(self.policy_weights.size())


def get_params(simulation: Simulation, fast: bool = False, inhibition: bool = False, theta_0: float = None
               ) -> tuple[int, int, list[float], torch.Tensor, torch.Tensor, torch.Tensor, dict, int]:
    """
    Initialize parameters depending on the simulation type.
    :param simulation: Simulation identifier
    :param fast: Whether to use the hyperparameters of the fast-learning regime instead of the biological ones
    :param inhibition: Whether the apical dendrites are inhibited (only for the bottom-up plasticity simulations)
    :param theta_0: Value of the theta_0 parameter (None: the value of the learning regime)
    :return: Tuple containing the total number of basal inputs, the number of distractor inputs, the probabilities of
    each distractor to be activated in a trial, the basal weights, an array of theta values for each pyramidal neuron,
    the apical synaptic weights, the hyperparameters of the simulation (see helper.get_hyperparameters) and the number
    of trials of the simulation.
    """
    hp = get_hyperparameters(simulation=simulation, fast=fast)
    if theta_0 is not None:
        hp['theta_0'] = theta_0
    n_task_stim = 3  # Number of task-relevant stimuli
    n_noise_stim = N_NOISE  # Number of distractor stimuli
    n_trials = get_n_trials(simulation=simulation, inhibition=inhibition)

    # Basal weights, plasticity thresholds (theta) and the Bernoulli distributions of the distractor stimuli
    n_stim = n_task_stim + n_noise_stim
    weights_apical = 0.05 + torch.rand(N_Z) * 0.1  # Sensory dendrite synaptic strengths
    noise_probs = [float(i / (n_noise_stim + 1)) for i in range(1, n_noise_stim + 1)]
    random_shuffle(noise_probs)

    # Lognormal basal weights: each neuron's afferents are L1-normalized to a total of INIT_SCALE, then clipped
    w_bas = torch.exp(INIT_SIGMA * torch.randn((n_stim, N_Z)))
    w_bas = torch.nn.functional.normalize(w_bas, p=1., dim=0)
    w_bas = torch.clamp(w_bas * INIT_SCALE, max=INIT_CLIP)
    probs = torch.tensor([1, 0.5, 0.5] + noise_probs) / N_TIME_STEPS
    thetas = hp['theta_0'] * torch.matmul(probs[:], w_bas) + BKG

    # Relabel the neurons so that neurons 0, 1 and 2 are those with the largest tone, texture 2 and texture 1 weight,
    # which the panels show as the cue, no-go and go neuron
    best = []
    for stim in (0, T2_IDX, T1_IDX):
        w = w_bas[stim].clone()
        w[best] = -1
        best.append(int(w.argmax()))
    perm = best + [i for i in range(N_Z) if i not in best]
    w_bas, weights_apical, thetas = w_bas[:, perm], weights_apical[perm], thetas[perm]

    return n_stim, n_noise_stim, noise_probs, w_bas, thetas.float(), weights_apical, hp, n_trials


def basal_rule(simulation: Simulation, x_in: torch.Tensor, x_som: torch.Tensor, x_ap: torch.Tensor, hp: dict,
               rule_state: dict) -> torch.Tensor:
    """
    Weight update of the basal synapses by the bottom-up plasticity rule of a simulation, before bounding. The rules
    have no explicit apical factor: the apical signal enters through the gain in the somatic activity x_som, so the
    rules also act at gain 1 (during apical inhibition).
    - BU_HEBB: x_in (x_som - theta_bas), with the basal threshold theta_bas = theta_bas_0 times the expected basal drive
      per time step (plus background)
    - BU_BCM (Bienenstock, Cooper & Munro, 1982): x_in x_som (x_som - theta_m), with a sliding threshold theta_m that
      relaxes towards x_som^2 / e0 with the time constant bcm_tau (in trials)
    - BU_CALCIUM (Shouval, Bear & Cooper, 2002; Graupner & Brunel, 2012): with the calcium level
      c = x_som (1 + ca_beta a), a being the apical activation rescaled to [0, 1], potentiation above the threshold
      theta_p and depression above ca_ltd_frac theta_p (smooth thresholds, depression rate ca_gamma_d). With
      ca_gamma_d >= 1 the net update of an active synapse is never positive (least depression above theta_p), so
      synapses only grow relative to others through the renormalization of the 'pre' or 'l1' bound
    :param simulation: Simulation identifier
    :param x_in: Binary vector of the currently active basal inputs (all zero at the outcome time step)
    :param x_som: Somatic activities
    :param x_ap: Apical activations
    :param hp: Hyperparameters of the simulation
    :param rule_state: Per-neuron thresholds of the rule (theta_bas, theta_m and e0, or theta_p), updated in place
    :return: Weight update (number of basal inputs x number of neurons)
    """
    if simulation == Simulation.BU_HEBB:
        post = x_som - rule_state['theta_bas']
    elif simulation == Simulation.BU_BCM:
        rule_state['theta_m'] += ((x_som.pow(2) / rule_state['e0'] - rule_state['theta_m'])
                                  / (hp['bcm_tau'] * N_TIME_STEPS))
        post = x_som * (x_som - rule_state['theta_m'])
    elif simulation == Simulation.BU_CALCIUM:
        theta_p = rule_state['theta_p']
        apical = torch.clamp((x_ap - 1. / MAX_GAIN) / (1 - 1. / MAX_GAIN), 0, 1)  # 0 at rest, 1 at maximal gain
        calcium = x_som * (1 + hp['ca_beta'] * apical)
        smooth = CA_SMOOTH * theta_p
        post = (torch.sigmoid((calcium - theta_p) / smooth)
                - hp['ca_gamma_d'] * torch.sigmoid((calcium - hp['ca_ltd_frac'] * theta_p) / smooth))
    else:
        raise ValueError(simulation)
    return torch.outer(hp['lr_bas'] * x_in, post)


def bound_basal_weights(w_bas: torch.Tensor, dw: torch.Tensor, bound: str, row_sums: torch.Tensor,
                        col_sums: torch.Tensor) -> torch.Tensor:
    """
    Apply an update to the basal weights and keep them in [0, 1].
    :param w_bas: Basal weights (number of basal inputs x number of neurons)
    :param dw: Weight update
    :param bound: 'soft' (the update is scaled by (1 - w) w), 'pre' (each stimulus keeps its initial total weight onto
    all neurons) or 'l1' (each neuron keeps its initial total basal input); the rescalings are followed by a clamp at 1
    :param row_sums: Initial total weight of each stimulus onto all neurons (for 'pre')
    :param col_sums: Initial total basal input of each neuron (for 'l1')
    :return: Updated basal weights
    """
    if bound == 'soft':
        return torch.clamp(w_bas + dw * (1 - w_bas) * w_bas, min=0, max=1)
    w_bas = torch.clamp(w_bas + dw, min=0)
    if bound == 'pre':
        return torch.clamp(w_bas * (row_sums / torch.clamp(w_bas.sum(1, keepdim=True), min=1e-12)), max=1)
    elif bound == 'l1':
        return torch.clamp(w_bas * (col_sums / torch.clamp(w_bas.sum(0, keepdim=True), min=1e-12)), max=1)
    raise ValueError(bound)


def simulate_seed(simulation: Simulation = Simulation.DEFAULT, seed: int = 0, fast: bool = False,
                  inhibition: bool = False, theta_0: float = None) -> dict:
    """
    Simulate learning of a go-nogo sensory discrimination task
    :param simulation: Simulation identifier
    :param seed: Pseudorandom seed
    :param fast: Whether to use the hyperparameters of the fast-learning regime instead of the biological ones
    :param inhibition: Whether to inhibit the apical dendrites for the first N_TRIALS trials (only for the bottom-up
    plasticity simulations)
    :param theta_0: theta_0 parameter (None: the value of the learning regime)
    :return: Dictionary containing the evolution of all the recorded variables of interest during the simulation
    """

    # Setting random seed
    rd_seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    # Initialization of basic params
    inhibited = is_inhibited(simulation=simulation, inhibition=inhibition)
    apical_active = not inhibited
    n_inputs, noise_inputs, noise_ps, w_bas, thetas, w_ap, hp, n_trials = get_params(
        simulation=simulation, fast=fast, inhibition=inhibition, theta_0=theta_0
    )
    n_z = N_Z
    tlr, learning_rate, lr_ap, plr = hp['lr_trace'], hp['lr_td'], hp['lr_ap'], hp['lr_policy']  # Learning rates

    # Bottom-up plasticity: per-neuron thresholds of the rule, set from the initial basal weights and the probability of
    # each stimulus per time step, and the initial weight totals kept by the bounds
    bu_plasticity = simulation in BU_SIMULATIONS
    w_bas_init = w_bas.clone()
    stim_probs = torch.tensor([1, 0.5, 0.5] + noise_ps) / N_TIME_STEPS
    rule_state = {}
    if simulation == Simulation.BU_HEBB:
        rule_state['theta_bas'] = (hp['theta_bas_0'] * torch.matmul(stim_probs, w_bas) + BKG).float()
    elif simulation == Simulation.BU_BCM:
        # The sliding threshold starts at theta, and e0 is set so that this is its fixed point at initialization
        x_som_sq = (torch.matmul(stim_probs, (w_bas + BKG) ** 2)
                    + torch.clamp(1 - stim_probs.sum(), min=0) * BKG ** 2)  # Expected x_som^2 per time step at gain 1
        rule_state['theta_m'] = thetas.clone()
        rule_state['e0'] = hp['bcm_e0_scale'] * x_som_sq / thetas
    elif simulation == Simulation.BU_CALCIUM:
        # Potentiation threshold relative to the strongest response the neuron can produce at gain 1
        rule_state['theta_p'] = hp['ca_theta_p'] * (w_bas.max(0).values + BKG)
    row_sums, col_sums = w_bas.sum(1, keepdim=True).clone(), w_bas.sum(0, keepdim=True).clone()

    # Relevant simulation time steps (first simulated time step starts with the tone cue)
    tone_t = 0  # Timing of the tone cue (in terms of simulated time steps index)
    texture_t = TEXTURE_T - TONE_T  # Timing of the texture (in terms of simulated time steps index)
    outcome_t = OUTCOME_T - TONE_T  # Timing of the outcome (in terms of simulated time steps index)
    n_timings = N_TIME_STEPS  # Number of time steps simulated per trial

    # Basal input initializations
    tx_idxs = torch.tensor([T1_IDX, T2_IDX])  # Indices for the texture selective neurons
    texture_xs = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    base_input_time_idxs = np.zeros(shape=n_inputs)  # Template for time steps at which each stimulus occurs
    base_input_time_idxs[0] = tone_t
    base_input_time_idxs[1] = texture_t
    base_input_time_idxs[2] = texture_t
    x_bas_background = torch.ones(n_z) * BKG
    preferred = w_bas.argmax(0)  # Preferred stimulus of each neuron (largest initial basal weight)

    # Initialize agent and state value estimator
    actor = Agent(nr_inputs=n_z, policy_lr=plr)  # Simulated mouse agent with action selection policy network
    x_pre_t = torch.zeros(outcome_t)  # Pre-synaptic inputs for sensory dendrites (time-step dependent)
    dw_weights = torch.zeros(n_z)

    # Initialization of empty objects in which various simulation variables will be recorded
    outcomes, correct_trials, den_out = tuple(nan_array((n_trials,)) for _ in range(3))
    d_sen_tx, d_sen_t1, d_sen_t2, x_pre_t_h, td_delta_h = tuple(nan_array((n_trials, n_timings)) for _ in range(5))
    w_ap_h, gain_h, x_som_texture_h = tuple(nan_array((n_trials, n_z), dtype=np.float32) for _ in range(3))
    z_h = np.zeros((n_trials, n_z))
    v_hat_h = np.zeros((n_trials, 1 + n_timings))
    snapshot_ts = sorted(set(list(range(0, n_trials, SNAPSHOT_EVERY)) + [n_trials - 1]))  # Trials of w_bas snapshots
    w_bas_h = np.zeros((len(snapshot_ts), n_inputs, n_z), dtype=np.float32)

    # Simulating trial after trial
    for j in range(n_trials):

        # Lift apical inhibition (in case it was present)
        if j == N_TRIALS and inhibited:
            apical_active = True

        # Initialize trial
        texture = np.random.choice([0, 1])  # Randomly sample a texture
        txt_in = texture_xs[texture]  # Get basal activation for sampled texture
        noise_in = torch.tensor([np.random.choice(a=[0, 1], p=[1-pr, pr]) for pr in noise_ps])  # Sample noise
        basal_input = torch.cat(tensors=(txt_in, noise_in), dim=0)  # Full basal activation vector
        input_time_idxs = torch.tensor(np.copy(base_input_time_idxs))  # Array for the timing of basal inputs
        input_time_idxs[3:] = torch.tensor(np.random.randint(low=0, high=outcome_t, size=noise_inputs))  # Random distractor timings
        input_time_idxs[:][basal_input < 0.5] = n_timings  # Timings of zero-inputs are set after the end of the trial

        # Storage variables
        dv_hat_before = torch.tensor([0.])  # State value change estimate at t-1
        v_hat_h[j, 0] = dv_hat_before.numpy()
        d_sen_tx[j, :outcome_t] = torch.mean(apical_transfer(torch.outer(x_pre_t, w_ap[tx_idxs])), dim=1)
        d_sen_t1[j, :outcome_t] = apical_transfer(x_pre_t * w_ap[T1_IDX])
        d_sen_t2[j, :outcome_t] = apical_transfer(x_pre_t * w_ap[T2_IDX])

        # Simulate each time step of the trial up to right before the action and outcome happen
        for t in range(outcome_t):

            # Simulate pyramidal neurons and compute state value estimate
            current_idxs = input_time_idxs == t  # Boolean mask of the currently non-zero basal inputs
            x_in = torch.zeros((n_inputs,))  # Basal input vector
            x_in[current_idxs] = 1  # Set currently active basal inputs to 1
            x_bas = torch.matmul(x_in, w_bas) + x_bas_background  # Basal activations
            x_ap = apical_transfer(w_ap * x_pre_t[t])  # Apical activations
            gains = gain(x_ap)  # Multiplicative apical gains
            x_som = gains * x_bas  # Somatic activations (also firing rate)
            actor.integrate_input(x_som)
            dv_hat_now = torch.dot(dw_weights, x_som)  # Estimate state value change

            # Update state value estimator, sensory dendrite afferent, sensory dendrite and basal synaptic weights
            dw_weights += learning_rate * dv_hat_now * z_h[j, :] * K_V
            w_ap = torch.clamp(w_ap + lr_ap * x_pre_t[t] * (x_som - thetas) * (1 - w_ap) * w_ap, min=0, max=1)
            if bu_plasticity:
                dw = basal_rule(simulation=simulation, x_in=x_in, x_som=x_som, x_ap=x_ap, hp=hp, rule_state=rule_state)
                w_bas = bound_basal_weights(w_bas=w_bas, dw=dw, bound=hp['bound'], row_sums=row_sums, col_sums=col_sums)
            x_pre_t[t] += tlr * (abs(dv_hat_now.item()) - x_pre_t[t])

            # Store variables of interest
            z_h[j, :] = GAMMA * z_h[j, :] + x_som.numpy()  # - BKG
            v_hat_h[j, 1 + t] = v_hat_h[j, t] / GAMMA + dv_hat_now.numpy()
            td_delta_h[j, t] = GAMMA * dv_hat_now
            driven = x_in[preferred] > 0  # Neurons whose preferred stimulus is present
            gain_h[j, driven.numpy()] = gains[driven].detach().clone().numpy()
            if t == texture_t:
                x_som_texture_h[j, :] = x_som.detach().clone().numpy()

        # Select the action, observe the reward obtained and update the policy network accordingly
        reward, licked, correct_trials[j] = actor.act(texture)

        # Record the outcome and the predicted reward
        if licked:
            r_pred = v_hat_h[j, outcome_t]
            if texture:
                outcomes[j] = Outcome.HIT.value
            else:
                outcomes[j] = Outcome.FA.value
        else:
            r_pred = 0.
            if texture:
                outcomes[j] = Outcome.MISS.value
            else:
                outcomes[j] = Outcome.CR.value

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
        gains = gain(x_ap)  # Multiplicative apical gains
        x_som = gains * x_bas_background  # Somatic activations (also firing rate)
        if bu_plasticity:
            # Without basal input the weights only change through the bound, but the thresholds of the rule see the
            # outcome-time somatic activity
            dw = basal_rule(simulation=simulation, x_in=torch.zeros(n_inputs), x_som=x_som, x_ap=x_ap.expand(n_z),
                            hp=hp, rule_state=rule_state)
            w_bas = bound_basal_weights(w_bas=w_bas, dw=dw, bound=hp['bound'], row_sums=row_sums, col_sums=col_sums)
        dw_weights += learning_rate * (reward - r_pred) * z_h[j, :] * (x_som - BKG)
        actor.update(reward=reward.item(), x_som=x_som)

        # Record variables of interest for post-simulation analysis
        v_hat_h[j, outcome_t+1] = v_hat_h[j, outcome_t] - r_pred
        if apical_active:
            d_sen_tx[j, outcome_t] = torch.mean(apical_transfer(sen_salience[tx_idxs]))
            d_sen_t1[j, outcome_t] = apical_transfer(sen_salience[T1_IDX])
            d_sen_t2[j, outcome_t] = apical_transfer(sen_salience[T2_IDX])
        else:
            d_sen_tx[j, outcome_t] = 0.
            d_sen_t1[j, outcome_t] = 0.
            d_sen_t2[j, outcome_t] = 0.
        w_ap_h[j, :] = w_ap
        x_pre_t_h[j, :-1] = x_pre_t
        x_pre_t_h[j, outcome_t] = abs(r_pred)
        td_delta_h[j, outcome_t] = reward - r_pred
        if j in snapshot_ts:
            w_bas_h[snapshot_ts.index(j)] = w_bas.numpy()

        # Update weights according to TD error elicited by resetting value estimate to zero at the end of the trial
        dw_weights += - learning_rate * torch.tensor(v_hat_h[j, outcome_t+1] * z_h[j, :]).float() * K_V

    # Stash all recorded simulation outcomes of interest into a single dictionary for saving
    result = {K_OUTCOME: outcomes,
              K_CORRECT_TRIALS: correct_trials,
              K_V_HAT: v_hat_h,
              K_TD_DELTA: td_delta_h,
              K_DENDRITE_SENSORY: d_sen_tx,
              K_DENDRITE_SENSORY_T1: d_sen_t1,
              K_DENDRITE_SENSORY_T2: d_sen_t2,
              K_DENDRITE_OUTCOME: den_out,
              K_GAIN: gain_h,
              K_W_AP: w_ap_h,
              K_TIMINGS: x_pre_t_h,
              K_W_BAS: w_bas_init.numpy(),
              K_W_BAS_FINAL: w_bas.numpy(),
              K_X_SOM_TXT: x_som_texture_h}
    if bu_plasticity:
        result[K_W_BAS_H] = w_bas_h
        result[K_W_BAS_T] = np.array(snapshot_ts)

    return result


def simulate_seeds(simulation: Simulation, fast: bool = False, inhibition: bool = False, theta_0: float = None):
    """
    Simulate multiple runs of the sensory discrimination task
    :param simulation: Simulation identifier
    :param fast: Whether to use the hyperparameters of the fast-learning regime instead of the biological ones
    :param inhibition: Whether to inhibit the apical dendrites for the first N_TRIALS trials (only for the bottom-up
    plasticity simulations)
    :param theta_0: theta_0 parameter of the theta_0 sweep of the default simulation (None: the value of the regime)
    """
    # String specifying the value of the theta_0 parameter used for the output log
    if theta_0 is not None:
        t0_str = f' with Theta_0 = {theta_0}'
    else:
        t0_str = ''

    # Simulate each seed and aggregate all the simulation outcomes in an array of entries for each seed
    results = []
    for s in range(N_SEEDS):
        print(f"\rSimulated {s}/{N_SEEDS} seeds...", end='')
        result = simulate_seed(seed=s, simulation=simulation, fast=fast, inhibition=inhibition, theta_0=theta_0)
        if theta_0 is not None:
            # The theta_0 sweep only keeps the trial outcomes, the basal weights and the final apical weights
            result = {K_OUTCOME: result[K_OUTCOME], K_CORRECT_TRIALS: result[K_CORRECT_TRIALS],
                      K_W_BAS: result[K_W_BAS], K_W_AP: result[K_W_AP][-1:, :]}
        results += [result]
    print(f"\rSimulation{t0_str} finished successfully.")

    # Save all the recorded variables of interest
    with open(get_path(simulation=simulation, fast=fast, inhibition=inhibition, theta_0=theta_0), 'wb') as handle:
        dump(results, handle, protocol=HIGHEST_PROTOCOL)


def save_trial_outcomes_matlab() -> None:
    """
    Save the arrays containing a bool for each trial describing whether the correct action was taken to matlab files
    """
    for simulation, fast, inhibition in get_runs():
        with open(get_path(simulation=simulation, fast=fast, inhibition=inhibition), 'rb') as handle:
            load_results = load(handle)
        save_results = {K_CORRECT_TRIALS: np.array([load_results[s][K_CORRECT_TRIALS] for s in range(N_SEEDS)])}
        savemat(SMITH_DIR / f'outcomes_{get_run_name(simulation, fast, inhibition)}.mat', save_results)
    print('Simulation results saved and ready to extract performance traces.')


def compute_smith_perf(n_workers: int = None) -> None:
    """
    Extract the performance traces, learning trials and expert trials from the trial outcomes in Smith/outcomes_*.mat
    with the Python port of the Smith et al. (2004) estimator and save them to Smith/performances.mat, as the Matlab
    script Smith/getperfs.m does.
    :param n_workers: Number of parallel processes (None: one per CPU)
    """
    names = [get_run_name(simulation, fast, inhibition) for simulation, fast, inhibition in get_runs()]
    correct_trials = [loadmat(str(SMITH_DIR / f'outcomes_{name}.mat'))[K_CORRECT_TRIALS] for name in names]
    with Pool(n_workers) as pool:
        analyses = pool.map(run_analysis, [c[s] for c in correct_trials for s in range(N_SEEDS)], chunksize=1)
    performances = {K_EXPERT_T: np.zeros((len(names), N_SEEDS)), K_LEARNING_T: np.zeros((len(names), N_SEEDS))}
    for i, name in enumerate(names):
        performances[f'perf_{name}'] = np.zeros((N_SEEDS, correct_trials[i].shape[1] + 1))
        for s in range(N_SEEDS):
            t_learn, t_expert, pmid, _ = analyses[i * N_SEEDS + s]
            performances[K_LEARNING_T][i, s] = t_learn
            performances[K_EXPERT_T][i, s] = t_expert
            performances[f'perf_{name}'][s, :] = pmid
    savemat(get_path(simulation=Simulation.SMITH_PERFORMANCE), performances)
    print('Performance traces have been extracted.')


def load_smith_perf() -> None:
    """
    Load the performances computed by compute_smith_perf() or by the Matlab script Smith/getperfs.m and save them into
    the results pickle files
    """

    # Load performance traces
    matlab_data = loadmat(str(get_path(simulation=Simulation.SMITH_PERFORMANCE)))

    # Load other results, insert performances into the results dictionary and save
    for i, (simulation, fast, inhibition) in enumerate(get_runs()):
        results_path = get_path(simulation=simulation, fast=fast, inhibition=inhibition)
        with open(results_path, 'rb') as handle:
            load_results = load(handle)
        for seed in range(N_SEEDS):
            for k in (K_EXPERT_T, K_LEARNING_T):  # Counted from 1 in Matlab, from 0 here (nan if not reached)
                t = matlab_data[k][i][seed]
                load_results[seed][k] = int(t) - 1 if np.isfinite(t) else np.nan
            load_results[seed][K_PERFORMANCE] = matlab_data[f'perf_{get_run_name(simulation, fast, inhibition)}'][seed]

        with open(results_path, "wb") as handle:
            dump(load_results, handle, protocol=HIGHEST_PROTOCOL)
    print('Extracted performance traces have been integrated into simulation results.')


def run():
    """
    Run all simulations, extract their performance traces and save their outcomes.
    """

    # Make sure the necessary directories exist
    makedirs()

    for simulation, fast, inhibition in get_runs():
        print(f'\nSimulating {get_run_name(simulation, fast, inhibition)}:')
        simulate_seeds(simulation=simulation, fast=fast, inhibition=inhibition)

    print('\nSimulating the theta_0 sweep:')
    for t0 in get_thetas_0():
        simulate_seeds(simulation=Simulation.DEFAULT, theta_0=t0)

    # Save the learning outcomes in a format that can be processed by Smith/getperfs.m, extract the performance traces
    # with the Python port of that script and integrate them into the results
    save_trial_outcomes_matlab()
    compute_smith_perf()
    load_smith_perf()
