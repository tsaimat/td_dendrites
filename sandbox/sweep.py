"""
Run bottom-up plasticity simulations over a grid of hyperparameters, in parallel over (config, seed).

Examples
    # one configuration, 3 seeds
    python -m sandbox.sweep --name oja_test --set rule=oja lr_bas=0.004 --seeds 3

    # cartesian grid over two hyperparameters, all other values from DEFAULT_HP
    python -m sandbox.sweep --name rules --grid '{"rule": ["soft_bounded", "l1_renormalized", "apical_gated"], "lr_bas": [0.004, 0.016]}'

    # quick smoke test
    python -m sandbox.sweep --name smoke --set n_trials=200 --seeds 2 --workers 2

Output: sandbox/results/<name>/<config>.pickle (list of per-seed result dicts) and summary.csv.
Then plot with:  python -m sandbox.plot <name>
"""
import argparse
import csv
import itertools
import json
import pickle
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
from sandbox.simulate import DEFAULT_HP, K_HP, simulate_seed_bu, summarize, SUMMARY_KEYS

RESULTS_DIR = Path(__file__).parent / 'results'


def parse_value(v: str):
    """Turn a command-line 'key=value' value into int, float, or string."""
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def config_name(hp: dict, varied: list[str]) -> str:
    return '_'.join(f"{k}-{hp[k]}" for k in varied) if varied else 'single'


def _worker(args):
    name, hp, seed = args
    torch.set_num_threads(1)
    try:
        return name, seed, simulate_seed_bu(hp, seed=seed)
    except Exception as e:  # a diverging rule must not kill the whole sweep
        return name, seed, f"FAILED: {type(e).__name__}: {e}"


def run_sweep(name: str, base: dict, grid: dict, seeds: int, workers: int, config_list: list[dict] = ()) -> Path:
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(grid)
    configs = {}
    for values in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
        hp = {**DEFAULT_HP, **base, **dict(zip(keys, values))}
        configs[config_name(hp, keys)] = hp
    if config_list:  # explicit, non-cartesian list of configurations (each a dict of overrides, named by them)
        configs = {'_'.join(f"{k}-{v}" for k, v in c.items()) or 'single': {**DEFAULT_HP, **base, **c} for c in config_list}
    jobs = [(n, hp, s) for n, hp in configs.items() for s in range(seeds)]
    print(f"{len(configs)} configurations x {seeds} seeds = {len(jobs)} runs on {workers} workers -> {out_dir}")

    results = {n: [None] * seeds for n in configs}
    with Pool(workers) as pool:
        for i, (n, s, res) in enumerate(pool.imap_unordered(_worker, jobs), 1):
            results[n][s] = res
            print(f"\r  {i}/{len(jobs)} done ({n}, seed {s})", end='', flush=True)
            if isinstance(res, str):
                print(f"\n  {n} seed {s} {res}")
    print()

    for n in configs:
        with open(out_dir / f"{n}.pickle", 'wb') as f:
            pickle.dump(results[n], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_dir / 'grid.json', 'w') as f:
        json.dump({'base': base, 'grid': grid, 'seeds': seeds}, f, indent=1)
    print_summary(write_summary(out_dir))
    return out_dir


def write_summary(out_dir: Path) -> list[dict]:
    """(Re)compute summary.csv of a sweep from its pickles (also usable after adding metrics to summarize())."""
    rows = []
    for p in sorted(out_dir.glob('*.pickle')):
        with open(p, 'rb') as f:
            results = pickle.load(f)
        good = [r for r in results if not isinstance(r, str)]
        hp = next((r[K_HP] for r in good), {})
        row = {'config': p.stem, **{k: hp.get(k) for k in DEFAULT_HP}, 'n_failed': len(results) - len(good)}
        per_seed = [summarize(r) for r in good]
        for k in SUMMARY_KEYS:
            vals = np.array([s[k] for s in per_seed], dtype=float)
            row[k] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan
            if k == 'diverged':
                row[k] = int(np.nansum(vals))  # number of seeds that diverged
            if k == 'expert_proxy':
                row['expert_proxy_n_not_reached'] = int(np.isnan(vals).sum())  # seeds that never hit the threshold
        rows.append(row)
    with open(out_dir / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return rows


def print_summary(rows: list[dict]) -> None:
    cols = [('diverged', '{:4.0f}'), ('expert_proxy', '{:7.0f}'), ('expert_proxy_n_not_reached', '{:4d}'), ('expert_after_lift', '{:7.0f}'),
            ('perf_inhibited_end', '{:6.3f}'), ('final_perf', '{:6.3f}'), ('task_frac_end', '{:6.3f}'),
            ('selectivity_end', '{:6.3f}'), ('det_tone', '{:4.0f}'), ('det_texture', '{:4.0f}'),
            ('det_distractor', '{:4.0f}'), ('w_bas_change', '{:9.2e}'), ('w_bas_sum_mean', '{:6.2f}')]
    short = {'diverged': 'ndiv', 'expert_proxy': 'expert', 'expert_proxy_n_not_reached': 'n_no', 'expert_after_lift': 'ex_lift',
             'perf_inhibited_end': 'p_inh', 'final_perf': 'p_end', 'task_frac_end': 'taskfr', 'selectivity_end': 'selec',
             'det_tone': 'dTon', 'det_texture': 'dTex', 'det_distractor': 'dDis',
             'w_bas_change': 'w_change', 'w_bas_sum_mean': 'w_sum'}
    width = max(len(r['config']) for r in rows)
    print(f"\n{'config':{width}s} fail " + ' '.join(f"{short[k]:>{len(f.format(0))}s}" for k, f in cols))
    for r in rows:
        cells = []
        for k, f in cols:
            v = r.get(k, float('nan'))  # older summaries lack columns added later
            cells.append(f.format(v) if not (isinstance(v, float) and np.isnan(v)) else f"{'-':>{len(f.format(0))}s}")
        print(f"{r['config']:{width}s} {r['n_failed']:4d} " + ' '.join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, help='sweep name (subfolder of sandbox/results)')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='override a default hyperparameter')
    ap.add_argument('--grid', default='{}', help='JSON dict {hp: [values]} or path to such a file')
    ap.add_argument('--configs', default='[]', help='JSON list of override dicts (non-cartesian alternative to --grid)')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    base = {k: parse_value(v) for k, v in (kv.split('=', 1) for kv in a.set)}
    def load_json(arg: str):
        return json.loads(arg) if arg.lstrip()[:1] in '[{' else json.load(open(arg))
    grid = load_json(a.grid)
    config_list = load_json(a.configs)
    unknown = set(base) | set(grid) | {k for c in config_list for k in c}
    unknown -= set(DEFAULT_HP)
    if unknown:
        raise SystemExit(f"unknown hyperparameters: {sorted(unknown)}; known: {sorted(DEFAULT_HP)}")
    run_sweep(a.name, base, grid, a.seeds, a.workers, config_list)


if __name__ == '__main__':
    main()
