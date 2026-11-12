"""Smith-scored scan of the gain-blind policy experiments (user, 2026-11-02: does bottom-up plasticity driven by the
apical signal support learning when the apical gain does not reach the policy?). Each job = (config, seed); the base is
control4 exactly as in the main code (neurons relabeled, gain recorded at the preferred stimulus), overridden by the
config. Expert and learning trials as in the main code (l5apical.simulations.get_performance: Smith et al. 2004 on the
whole run, counted from 0, nan if not reached). One JSON line per job, appended as jobs finish.
Usage: python -m sandbox.scan_gainless <out.jsonl> '<json {name: hp overrides}>' [seeds, e.g. 0-9] [n_workers]"""
import json, sys, os
from multiprocessing import Pool
import numpy as np, torch
from l5apical.helper import *
from l5apical.simulations import get_performance
from sandbox.simulate import simulate_seed_bu, K_DIVERGED, K_W_BAS_H

BASE = dict(n_z=1000, n_noise=30, init='lognormal', init_sigma=2, init_norm='neuron', init_scale=1.89, init_clip=1,
            lr_trace=0.256, rule='none', theta_0=2.35, lr_td=0.00044, lr_policy=0.004, lr_ap=0.112,
            sort_neurons=True, gain_at_preferred=True, n_trials=N_TRIALS)


def job(args):
    name, over, seed = args
    torch.set_num_threads(1)
    hp = {**BASE, **over}
    r = simulate_seed_bu(hp, seed=seed)
    c = np.asarray(r[K_CORRECT_TRIALS], float)
    perf = get_performance(c)
    n = len(c)
    w0, w1 = r[K_W_BAS_H][0], r[K_W_BAS_H][-1]
    recruited = r[K_W_AP][-1] > 0.5
    tex = w1[[T1_IDX, T2_IDX]]
    out = dict(name=name, over=over, seed=seed, n_trials=n,
               expert=None if np.isnan(perf[K_EXPERT_T]) else int(perf[K_EXPERT_T]),
               learning=None if np.isnan(perf[K_LEARNING_T]) else int(perf[K_LEARNING_T]),
               diverged=r[K_DIVERGED],
               perf_200=[float(np.mean(np.nan_to_num(c[a:a + 200], nan=0.))) for a in range(0, n, 200)],
               t80=(int(np.argmax(perf[K_PERFORMANCE][1:] >= 0.8)) if (perf[K_PERFORMANCE][1:] >= 0.8).any()
                    else None),  # first trial with the Smith estimate at 0.8 or more
               final_perf=float(np.nanmean(np.nan_to_num(c, nan=0.)[-200:])),
               lick_200=[float(np.mean(np.isin(r[K_OUTCOME][a:a + 200], (Outcome.HIT.value, Outcome.FA.value))))
                         for a in range(0, n, 200)],
               recruited=float(np.mean(recruited)),
               w_change=float(np.abs(w1 - w0).sum(0).mean()),
               tex_detectors=float(np.mean(tex.max(0) > 0.5)),
               tex_weight_recruited=float(tex[:, recruited].sum(0).mean()) if recruited.any() else None,
               tone_weight_recruited=float(w1[0, recruited].mean()) if recruited.any() else None)
    return out


def parse_seeds(s):
    a, b = s.split('-')
    return list(range(int(a), int(b) + 1))


if __name__ == '__main__':
    path, variants = sys.argv[1], json.loads(sys.argv[2])
    seeds = parse_seeds(sys.argv[3]) if len(sys.argv) > 3 else list(range(10))
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    done = set()
    if os.path.exists(path):
        for line in open(path):
            d = json.loads(line)
            done.add((d['name'], d['seed']))
    jobs = [(n, o, s) for n, o in variants.items() for s in seeds if (n, s) not in done]
    jobs.sort(key=lambda j: -j[1].get('n_trials', N_TRIALS))  # long jobs first for load balance
    with Pool(n_workers) as pool, open(path, 'a') as f:
        for d in pool.imap_unordered(job, jobs):
            f.write(json.dumps(d) + '\n')
            f.flush()
    print('SCAN DONE', path, flush=True)
