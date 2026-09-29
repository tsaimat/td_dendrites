"""Causal ablation for us / burst: same rule, but with the flag the running average (g_bar / p_bar) ignores the outcome
time step. Prints per 300-trial block the mean gain, the running average and the fraction of neurons with a positive
post-synaptic factor at the tone, texture and outcome steps, plus expert proxy and detector counts per seed.
Not a candidate mechanism (the user considers step-specific baselines unbiological), only a diagnostic. Usage:
    python -m sandbox.diag_gated_ablation '<hp json>' <n_seeds> [ablate]
"""
import sys, json, numpy as np, torch
from l5apical.helper import *
import sandbox.rules as R
from sandbox.simulate import simulate_seed_bu, K_W_BAS_H, expert_trial_proxy
hp = json.loads(sys.argv[1]); n_seeds = int(sys.argv[2]); ablate = len(sys.argv) > 3
base = R.RULES[hp['rule']]; key = 'g_bar' if hp['rule'] == 'us' else 'p_bar'
tone_t, texture_t, outcome_t = 0, TEXTURE_T - TONE_T, OUTCOME_T - TONE_T
LOG = {}
def wrapped(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp_, state):
    t = LOG['t']
    saved = state[key].clone() if (ablate and t == outcome_t and key in state) else None
    dw = base(w_bas, x_in, x_som, x_bas, x_ap, gains, thetas, hp_, state)
    if saved is not None:
        state[key].copy_(saved)
    if t in (tone_t, texture_t, outcome_t):
        LOG.setdefault(t, []).append((gains.mean().item(), state[key].mean().item(),
                                      ((gains if key == 'g_bar' else x_ap + hp_.get('burst_kappa', 0.) * x_som) - state[key] > 0).float().mean().item()))
    LOG['t'] = (t + 1) % (outcome_t + 1)
    return dw
R.RULES[hp['rule']] = wrapped
ex = []
for seed in range(n_seeds):
    LOG.clear(); LOG['t'] = 0
    r = simulate_seed_bu(hp, seed=seed)
    ex.append(expert_trial_proxy(r[K_CORRECT_TRIALS]))
    w = r[K_W_BAS_H][-1]
    det = lambda rows: int((w[rows].max(0) > 0.5).sum()) if np.ndim(w[rows]) == 2 else int((w[rows] > 0.5).sum())
    print(f'seed {seed}: expert {ex[-1]:.0f} diverged {r.get("diverged_at")}  detectors tone {det(0)} texture {det(slice(1,3))} distractor {det(slice(3,None))}', flush=True)
    if seed == 0:
        for name, t in (('tone', tone_t), ('texture', texture_t), ('outcome', outcome_t)):
            a = np.array(LOG[t])
            print(f'   {name:8s} step: per 300-trial block  mean gain / mean {key} / frac neurons with positive factor')
            for b in range(0, len(a), 300):
                blk = a[b:b+300].mean(0); print(f'      {b:5d}: {blk[0]:.3f} {blk[1]:.3f} {blk[2]:.2f}')
print('ablate' if ablate else 'normal', 'median expert', np.nanmedian(ex), 'n learned', int(np.isfinite(ex).sum()), '/', n_seeds)
