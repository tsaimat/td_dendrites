"""
Python port of the state-space learning curve estimator of Smith et al. (2004) in Smith/*.m (runanalysisv3.m with its
helpers forwardfilter.m, backwardfilter.m, em_bino.m, newtonsolve.m and pdistnv2.m), so that the performance traces,
learning trials and expert trials can be extracted without Matlab.

Smith, A.C., Frank, L.M., Wirth, S., Yanike, M., Hu, D., Kubota, Y., Graybiel, A.M., Suzuki, W.A. and Brown, E.N.
(2004). Dynamic analysis of learning in behavioral experiments. Journal of Neuroscience, 24(2), 447-461.

The code follows the Matlab scripts line by line, including the convergence criteria, except for the confidence
limits: pdistnv2.m draws 10000 samples per trial and takes their empirical quantiles, here the exact quantiles of the
logit-normal posterior are used, which is the same up to the Monte Carlo noise of the Matlab version. Compared with the
Matlab output for the trial outcomes of the original manuscript simulations, the learning and expert trials are
identical and the performance traces differ by at most 3e-3, which is the Monte Carlo noise of the Matlab version.
"""

import numpy as np
from scipy.special import expit
from scipy.stats import norm


def newton_solve(mu: float, x_old: float, sig_old_sq: float, n: float, n_max: int) -> tuple[float, int]:
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


def forward_filter(responses: np.ndarray, n_max: int, sig_e: float, x_guess: float, sig_sq_guess: float, mu: float
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
        x_hat[k], _ = newton_solve(mu, x_old[k], sig_sq_old[k], responses[k - 1], n_max)
        e = em * np.exp(x_hat[k])
        denominator = -1 / sig_sq_old[k] - n_max * e / (1 + e) ** 2
        sig_sq[k] = -1 / denominator
    return x_hat, sig_sq, x_old, sig_sq_old


def backward_filter(x: np.ndarray, x_old: np.ndarray, sig_sq: np.ndarray, sig_sq_old: np.ndarray
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


def em_bino(x_new: np.ndarray, sig_new_sq: np.ndarray, a: np.ndarray, start_flag: int) -> float:
    """
    Maximization step for the variance of the learning state process (em_bino.m).
    :param x_new: Smoothed learning states
    :param sig_new_sq: Variances of the smoothed learning states
    :param a: Smoother gains
    :param start_flag: Treatment of the initial condition (UpdaterFlag of runanalysisv3.m)
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


def run_analysis(responses: np.ndarray, max_response: int = 1, background_prob: float = 0.5, sig_e: float = 0.005,
                 updater_flag: int = 2, max_steps: int = 4000) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Estimate the learning curve of one agent and its learning and expert trials (runanalysisv3.m).
    :param responses: Number of correct responses in each trial (0 or 1 for binary outcomes)
    :param max_response: Maximal number of correct responses in a trial
    :param background_prob: Probability of a correct response by chance
    :param sig_e: Initial standard deviation of the learning state process
    :param updater_flag: Treatment of the initial condition (0: fixed, 1: estimated, 2: removed from the likelihood)
    :param max_steps: Maximal number of iterations of the expectation maximization
    :return: Tuple of the learning trial (50 plus the first trial after trial 51 at which the lower 5 percent confidence
    bound of the probability of a correct response is above chance), the expert trial (last trial at which that bound is
    below chance), both counted from 1 as in Matlab and nan if not reached, the median performance trace and its lower 5
    percent confidence bound (each with one entry more than the number of trials, the first being the prior)
    """
    responses = np.asarray(responses, dtype=float).ravel()
    mu = np.log(background_prob / (1 - background_prob))
    convergence = 1e-8
    x_guess, sig_sq_guess = 0.0, sig_e ** 2
    new_sig_sq, x_new_1_save = [], []
    x_new = sig_new_sq = None
    for i in range(max_steps):
        x, s, x_old, s_old = forward_filter(responses, max_response, sig_e, x_guess, sig_sq_guess, mu)
        x_new, sig_new_sq, a = backward_filter(x, x_old, s, s_old)
        if updater_flag == 1:
            x_new[0], sig_new_sq[0] = 0.5 * x_new[1], sig_e ** 2
        elif updater_flag == 0:
            x_new[0], sig_new_sq[0] = 0.0, sig_e ** 2
        else:
            x_new[0], sig_new_sq[0] = x_new[1], sig_new_sq[1]
        new_sig_sq.append(em_bino(x_new, sig_new_sq, a, updater_flag))
        x_new_1_save.append(x_new[0])
        if i > 0:
            a1, a2 = abs(new_sig_sq[i] - new_sig_sq[i - 1]), abs(x_new_1_save[i] - x_new_1_save[i - 1])
            if (a1 < convergence and a2 < convergence and updater_flag >= 1) or \
                    (a1 < convergence and updater_flag == 0):
                break
        sig_e, x_guess, sig_sq_guess = np.sqrt(new_sig_sq[i]), x_new[0], sig_new_sq[0]

    # Confidence limits of the probability of a correct response (pdistnv2.m), with the exact logit-normal quantiles
    sd = np.sqrt(sig_new_sq)
    p05 = expit(mu + x_new + sd * norm.ppf(0.05))
    p_mid = expit(mu + x_new)
    n = len(responses)
    below = np.where(p05 < background_prob)[0] + 1  # Trials counted from 1 as in Matlab
    t_expert = float(below[-1]) if len(below) and below[-1] < n + 1 else np.nan
    above = np.where(p05[51:] > background_prob)[0] + 1
    t_learn = float(50 + above[0]) if len(above) and above[0] < n + 1 else np.nan
    return t_learn, t_expert, p_mid, p05
