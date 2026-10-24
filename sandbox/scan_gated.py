"""Smith-scored scan of the rules with an explicit apical factor (gated_hebb, burst, us) on top of control4 (user,
2026-10-18). Same jobs and expert-trial definition as sandbox.scan_smith (one JSON line per (config, seed, protocol),
Smith on the full correct-trial series with NaN trials of diverged runs counted as errors, expert trial minus 1, after
the lift minus the inhibited trials), with the control4 settings of s1_control4/finalists.json as the base, the
inhibition protocol of the main code (N_TRIALS_AP_INH trials, the first N_TRIALS inhibited) and weight metrics.
Usage: python -m sandbox.scan_gated <out.jsonl> '<json {name: overrides of the base}>' [seeds, e.g. 0-9]
       [both|fresh|inh] [trace]   (trace: also store each run's Smith performance trace, rounded to 4 decimals)"""
import json, sys, os
from multiprocessing import Pool
import numpy as np, torch
from l5apical.helper import N_TRIALS, N_TRIALS_AP_INH, K_CORRECT_TRIALS, K_W_AP, K_W_BAS, T1_IDX, T2_IDX
from sandbox.simulate import simulate_seed_bu, K_DIVERGED, K_W_BAS_H, K_W_BAS_T
from sandbox.smith import run_analysis
from sandbox.scan_smith import cats, parse_seeds

BASE = json.load(open(os.path.join(os.path.dirname(__file__), 'results', 's1_control4', 'finalists.json')))['control4']
DETECTOR = 0.5  # as DETECTOR_THRESHOLD of l5apical.panels: a texture detector has a weight above it from either texture
TRACE = 'trace' in sys.argv[5:]


def weight_metrics(w: np.ndarray, w_ref: np.ndarray, tag: str) -> dict:
    """Percent of neurons that are texture (tone) detectors and of those with both texture weights above the detector
    threshold, the task inputs' (tone and textures) percentage of all basal weight, the mean number of inputs above
    0.05 per neuron, mean total input and largest weight per neuron, percent of neurons without any weight above 0.05,
    and mean |w - w_ref| over all synapses (w_ref: after trial 0)."""
    return {f'det_texture_{tag}': float(100 * np.mean((w[T1_IDX] > DETECTOR) | (w[T2_IDX] > DETECTOR))),
            f'det_both_{tag}': float(100 * np.mean((w[T1_IDX] > DETECTOR) & (w[T2_IDX] > DETECTOR))),
            f'task_share_{tag}': float(100 * w[:3].sum() / w.sum()),
            f'n_inputs_{tag}': float((w > 0.05).sum(0).mean()),
            f'det_tone_{tag}': float(100 * np.mean(w[0] > DETECTOR)),
            f'w_total_{tag}': float(w.sum(0).mean()), f'w_max_{tag}': float(w.max(0).mean()),
            f'no_input_{tag}': float(100 * np.mean(w.max(0) <= 0.05)),
            f'dw_abs_{tag}': float(np.abs(w - w_ref).mean())}


def job(args):
    name, over, seed, prot = args
    torch.set_num_threads(1)
    inh = N_TRIALS if prot == 'inh' else 0
    hp = {**BASE, **over, 'n_trials': N_TRIALS_AP_INH if inh else N_TRIALS, 'n_trials_inhibited': inh,
          'store_full': True}
    r = simulate_seed_bu(hp, seed=seed)
    c = np.nan_to_num(np.asarray(r[K_CORRECT_TRIALS], float), nan=0.)
    _, t_exp, perf, _ = run_analysis(c)
    expert = (t_exp - 1 - inh) if np.isfinite(t_exp) else None
    out = dict(name=name, over=over, seed=seed, prot=prot, expert=expert, diverged=r[K_DIVERGED],
               final_perf=float(c[-200:].mean()))
    if TRACE:
        out['perf'] = np.round(np.asarray(perf, float), 4).tolist()
    if r[K_DIVERGED] is None:
        w_h, t_h = np.asarray(r[K_W_BAS_H]), np.asarray(r[K_W_BAS_T])
        w_end = np.asarray(r[K_W_BAS], float)
        out.update(weight_metrics(w_end, w_h[0], 'end'))
        if inh:
            out.update(weight_metrics(w_h[int(np.searchsorted(t_h, inh)) - 1], w_h[0], 'lift'))
        c0, c1 = cats(r, inh, inh + 100), cats(r, inh + N_TRIALS - 100, inh + N_TRIALS)
        T = np.array([[np.sum((c0 == a) & (c1 == b)) for b in range(4)] for a in range(4)]) / len(c0)
        out.update(trans=T.tolist(), full_gain=float(np.mean(r[K_W_AP][-1] > 0.5)))
    return out


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
