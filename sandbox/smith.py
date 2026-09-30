"""
Python port of the Smith et al. (2004) state-space learning-curve estimator in Smith/*.m (runanalysisv3 with its
helpers forwardfilter, backwardfilter, em_bino, newtonsolve, pdistnv2), so that the performance traces, expert
trials and learning trials of the manuscript panels can be produced without Matlab.

Differences from the Matlab code: the confidence limits are the exact quantiles of the logit-normal posterior
(pdistnv2.m draws 10000 samples per trial and takes empirical quantiles, which is the same up to sampling noise);
everything else follows the Matlab code line by line, including the convergence criteria and the definitions
    t_expert = last trial at which the lower 5 percent bound is below chance (NaN if it never rises above it)
    t_learn  = 50 + first trial after trial 51 at which the lower bound is above chance.
Both are 1-based trial indices as in Matlab, and the traces have n_trials + 1 entries (trial 0 is the prior).
"""
import numpy as np
from scipy.special import expit
from scipy.stats import norm


def _newton(mu, xold, sigoldsq, n, nmax):
    """newtonsolve.m: posterior mode of the learning state for one trial."""
    em = np.exp(mu)
    for x0 in (None, -1.0, 1.0):
        it = xold + sigoldsq * (n - nmax * em * np.exp(xold) / (1 + em * np.exp(xold))) if x0 is None else x0
        for _ in range(100):
            e = em * np.exp(it)
            g = xold + sigoldsq * (n - nmax * e / (1 + e)) - it
            gp = -nmax * sigoldsq * e / (1 + e) ** 2 - 1
            new = it - g / gp
            if abs(new - it) < 1e-10:
                return new, 0
            it = new
    return it, 1


def forward_filter(responses, nmax, sig_e, xguess, sigsqguess, mu):
    """forwardfilter.m. Returns x{k|k}, sigsq{k|k}, x{k|k-1}, sigsq{k|k-1}, each with K + 1 entries."""
    K = len(responses)
    xhat, sigsq = np.zeros(K + 1), np.zeros(K + 1)
    xold, sigsqold = np.zeros(K + 1), np.zeros(K + 1)
    xhat[0], sigsq[0] = xguess, sigsqguess
    em = np.exp(mu)
    for k in range(1, K + 1):
        xold[k] = xhat[k - 1]
        sigsqold[k] = sigsq[k - 1] + sig_e ** 2
        xhat[k], _ = _newton(mu, xold[k], sigsqold[k], responses[k - 1], nmax)
        e = em * np.exp(xhat[k])
        denom = -1 / sigsqold[k] - nmax * e / (1 + e) ** 2
        sigsq[k] = -1 / denom
    return xhat, sigsq, xold, sigsqold


def backward_filter(x, xold, sigsq, sigsqold):
    """backwardfilter.m. Returns x{k|K}, sigsq{k|K} and A{k} (A[0] unused, as in Matlab where A(1) is never set)."""
    T = len(x)
    xnew, signewsq, A = np.zeros(T), np.zeros(T), np.zeros(T)
    xnew[T - 1], signewsq[T - 1] = x[T - 1], sigsq[T - 1]
    for i in range(T - 2, 0, -1):
        A[i] = sigsq[i] / sigsqold[i + 1]
        xnew[i] = x[i] + A[i] * (xnew[i + 1] - xold[i + 1])
        signewsq[i] = sigsq[i] + A[i] ** 2 * (signewsq[i + 1] - sigsqold[i + 1])
    return xnew, signewsq, A


def em_bino(xnew, signewsq, A, startflag):
    """em_bino.m: M step for the learning-state process variance."""
    M = len(xnew)
    xnewt, xnewtm1, signewsqt = xnew[2:M], xnew[1:M - 1], signewsq[2:M]
    covcalc = signewsqt * A[1:M - 1]  # Matlab A(k) is A[k-1] here; A(2:end) pairs signewsq(k) with A(k-1)
    term1 = np.sum(xnewt ** 2) + np.sum(signewsqt)
    term2 = np.sum(covcalc) + np.sum(xnewt * xnewtm1)
    if startflag == 1:
        term3, term4 = 1.5 * xnew[1] ** 2 + 2.0 * signewsq[1], xnew[-1] ** 2 + signewsq[-1]
    elif startflag == 0:
        term3, term4 = 2 * xnew[1] ** 2 + 2 * signewsq[1], xnew[-1] ** 2 + signewsq[-1]
    else:
        term3, term4 = xnew[1] ** 2 + 2 * signewsq[1], xnew[-1] ** 2 + signewsq[-1]
        M = M - 1
    return (2 * (term1 - term2) + term3 - term4) / M


def run_analysis(responses, max_response: int = 1, background_prob: float = 0.5, sig_e: float = 0.005,
                 updater_flag: int = 2, max_steps: int = 4000, verbose: bool = False):
    """runanalysisv3.m. Returns (t_learn, t_expert, pmid, p05) with pmid and p05 of length n_trials + 1."""
    responses = np.asarray(responses, dtype=float).ravel()
    mu = np.log(background_prob / (1 - background_prob))
    cvg = 1e-8
    xguess, sigsqguess = 0.0, sig_e ** 2
    newsigsq, xnew1save = [], []
    xnew = signewsq = None
    for i in range(max_steps):
        x, s, xold, sold = forward_filter(responses, max_response, sig_e, xguess, sigsqguess, mu)
        xnew, signewsq, A = backward_filter(x, xold, s, sold)
        if updater_flag == 1:
            xnew[0], signewsq[0] = 0.5 * xnew[1], sig_e ** 2
        elif updater_flag == 0:
            xnew[0], signewsq[0] = 0.0, sig_e ** 2
        else:
            xnew[0], signewsq[0] = xnew[1], signewsq[1]
        newsigsq.append(em_bino(xnew, signewsq, A, updater_flag))
        xnew1save.append(xnew[0])
        if i > 0:
            a1, a2 = abs(newsigsq[i] - newsigsq[i - 1]), abs(xnew1save[i] - xnew1save[i - 1])
            if (a1 < cvg and a2 < cvg and updater_flag >= 1) or (a1 < cvg and updater_flag == 0):
                if verbose:
                    print(f'converged after {i + 1} steps')
                break
        sig_e, xguess, sigsqguess = np.sqrt(newsigsq[i]), xnew[0], signewsq[0]
    else:
        if verbose:
            print(f'failed to converge after {max_steps} steps')
    # pdistnv2.m with exact quantiles of the logit-normal posterior
    sd = np.sqrt(signewsq)
    p05 = expit(mu + xnew + sd * norm.ppf(0.05))
    pmid = expit(mu + xnew)
    n = len(responses)
    below = np.where(p05 < background_prob)[0] + 1  # 1-based as in Matlab
    t_expert = float(below[-1]) if len(below) and below[-1] < n + 1 else np.nan
    above = np.where(p05[51:] > background_prob)[0] + 1
    t_learn = float(50 + above[0]) if len(above) and above[0] < n + 1 else np.nan
    return t_learn, t_expert, pmid, p05


if __name__ == '__main__':  # validate against the Matlab output shipped with the repository
    import sys, time
    from scipy.io import loadmat
    from l5apical.helper import SMITH_DIR
    perf = loadmat(SMITH_DIR / 'performances.mat')
    out = loadmat(SMITH_DIR / 'outcomes_default.mat')
    n_check = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for s in range(n_check):
        t0 = time.time()
        t_learn, t_expert, pmid, p05 = run_analysis(out['correct_trials'][s])
        ref = perf['perf_default'][s]
        print(f"seed {s}: expert {t_expert:.0f} (Matlab {perf['expert_t'][0, s]:.0f}), learning {t_learn:.0f} "
              f"(Matlab {perf['learning_t'][0, s]:.0f}), max |pmid - Matlab| {np.abs(pmid - ref).max():.2e}, {time.time() - t0:.1f} s")
