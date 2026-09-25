"""One-seed diagnostic of the apical loop: TD-error profile, apical afferent trace x_pre_t, apical weights, gains and
texture weights at the start and end of a run. Usage (hp as JSON, optional seed):
    python -m sandbox.diag '{"rule": "gated_hebb", "n_z": 20, "init_power": 12}' 0
"""
import sys, json, numpy as np, torch
from l5apical.helper import *
from sandbox.simulate import simulate_seed_bu, K_W_BAS_H
hp = json.loads(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
r = simulate_seed_bu({**hp, 'store_full': True}, seed=seed)
n = len(r[K_CORRECT_TRIALS]) if K_CORRECT_TRIALS in r else None
def blk(a, k=300): return np.nanmean(np.asarray(a, float)[:k], 0), np.nanmean(np.asarray(a, float)[-k:], 0)
print('diverged', r.get('diverged_at'), 'perf first/last 300:', [round(float(np.nanmean(np.asarray(r[k], float)[s])), 3) for k in [K_CORRECT_TRIALS] for s in (slice(0, 300), slice(-300, None))])
a, b = blk(r[K_TIMINGS]); print('x_pre_t   first', np.round(a, 3), '\n          last ', np.round(b, 3))
a, b = blk(r[K_TD_DELTA]); print('|td|      first', np.round(np.abs(a), 3), '\n          last ', np.round(np.abs(b), 3))
a, b = blk(r[K_W_AP]); print('w_ap mean first/last', round(float(a.mean()), 3), round(float(b.mean()), 3), ' max', round(float(b.max()), 3))
a, b = blk(r[K_GAIN]); print('gain mean first/last', round(float(a.mean()), 3), round(float(b.mean()), 3), ' max', round(float(b.max()), 3))
w0, w1 = r[K_W_BAS_H][0], r[K_W_BAS_H][-1]
print('w_bas texture rows (T2, T1) init max', np.round(w0[1:3].max(1), 3), '\n                     final max', np.round(w1[1:3].max(1), 3))
print('total change', round(float(np.abs(w1 - w0).mean()), 5))
