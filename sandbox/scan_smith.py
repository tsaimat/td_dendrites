"""Smith-scored scan. Each job = (config, seed, protocol). Expert trials follow the pipeline exactly: sandbox.smith.run_analysis
(validated port of runanalysisv3.m) on the full correct-trial series (NaN trials of diverged runs count as errors), expert
trial minus 1 as in load_smith_perf; after-lift value = that minus N_TRIALS. One JSON line per job, appended as jobs finish.
Usage: python -m sandbox.scan_smith <out.jsonl> '<json {name: hp overrides}>' [seeds, e.g. 0-9] [protocols: both|fresh|inh]"""
import json, sys, os
from multiprocessing import Pool
import numpy as np, torch
from l5apical.helper import *
from l5apical.panels import get_texture_specific
from sandbox.simulate import simulate_seed_bu, K_DIVERGED
from sandbox.smith import run_analysis

BASE = dict(rule='none', n_z=1000, n_noise=30, init='lognormal', init_sigma=2, init_norm='neuron', init_scale=1.89, init_clip=1,
            theta_0=2.25, lr_policy=0.004, lr_td=0.0005, lr_ap=0.128, lr_trace=0.256)

def cats(r, a, b):
    x = r[K_X_SOM_TXT][a:b]
    x1 = np.nanmean(get_texture_specific(True, r[K_OUTCOME], r[K_X_SOM_TXT])[a:b], 0)
    x2 = np.nanmean(get_texture_specific(False, r[K_OUTCOME], r[K_X_SOM_TXT])[a:b], 0)
    resp = x.mean(0) > 0.1; d = x1 - x2
    return np.where(~resp, 0, np.where(np.abs(d) < 0.2, 1, np.where(d > 0, 2, 3)))

def job(args):
    name, over, seed, prot = args
    torch.set_num_threads(1)
    inh = N_TRIALS if prot == 'inh' else 0
    hp = {**BASE, **over, 'n_trials': N_TRIALS + inh, 'n_trials_inhibited': inh, 'store_full': True}
    r = simulate_seed_bu(hp, seed=seed)
    c = np.nan_to_num(np.asarray(r[K_CORRECT_TRIALS], float), nan=0.)
    _, t_exp, _, _ = run_analysis(c)
    expert = (t_exp - 1 - inh) if np.isfinite(t_exp) else None
    out = dict(name=name, over=over, seed=seed, prot=prot, expert=expert, diverged=r[K_DIVERGED],
               final_perf=float(c[-200:].mean()))
    if r[K_DIVERGED] is None:
        c0, c1 = cats(r, inh, inh + 100), cats(r, inh + N_TRIALS - 100, inh + N_TRIALS)
        T = np.array([[np.sum((c0 == a) & (c1 == b)) for b in range(4)] for a in range(4)]) / len(c0)
        out.update(trans=T.tolist(), full_gain=float(np.mean(r[K_W_AP][-1] > 0.5)))
    return out

def parse_seeds(s):
    a, b = s.split('-'); return list(range(int(a), int(b) + 1))

if __name__ == '__main__':
    path, variants = sys.argv[1], json.loads(sys.argv[2])
    seeds = parse_seeds(sys.argv[3]) if len(sys.argv) > 3 else list(range(10))
    prots = {'both': ['fresh', 'inh'], 'fresh': ['fresh'], 'inh': ['inh']}[sys.argv[4] if len(sys.argv) > 4 else 'both']
    done = set()
    if os.path.exists(path):
        for l in open(path):
            d = json.loads(l); done.add((d['name'], d['seed'], d['prot']))
    jobs = [(n, o, s, p) for p in prots for n, o in variants.items() for s in seeds if (n, s, p) not in done]
    jobs.sort(key=lambda j: j[3] != 'inh')  # long jobs first for load balance
    with Pool(8) as pool, open(path, 'a') as f:
        for d in pool.imap_unordered(job, jobs):
            f.write(json.dumps(d) + '\n'); f.flush()
    print('SCAN DONE', path, flush=True)
