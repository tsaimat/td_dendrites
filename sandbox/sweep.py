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
from sandbox.simulate import DEFAULT_HP, simulate_seed_bu, summarize, SUMMARY_KEYS

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


def run_sweep(name: str, base: dict, grid: dict, seeds: int, workers: int) -> Path:
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(grid)
    configs = {}
    for values in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
        hp = {**DEFAULT_HP, **base, **dict(zip(keys, values))}
        configs[config_name(hp, keys)] = hp
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

    rows = []
    for n, hp in configs.items():
        with open(out_dir / f"{n}.pickle", 'wb') as f:
            pickle.dump(results[n], f, protocol=pickle.HIGHEST_PROTOCOL)
        good = [r for r in results[n] if not isinstance(r, str)]
        row = {'config': n, **{k: hp[k] for k in DEFAULT_HP}, 'n_failed': seeds - len(good)}
        per_seed = [summarize(r) for r in good]
        for k in SUMMARY_KEYS:
            vals = np.array([p[k] for p in per_seed], dtype=float)
            row[k] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan
            if k == 'expert_proxy':
                row['expert_proxy_n_not_reached'] = int(np.isnan(vals).sum())  # seeds that never hit the threshold
        rows.append(row)
    with open(out_dir / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(out_dir / 'grid.json', 'w') as f:
        json.dump({'base': base, 'grid': grid, 'seeds': seeds}, f, indent=1)

    print(f"\n{'config':40s} {'failed':>6s} {'expert_proxy':>12s} {'final_perf':>10s} {'task_frac end':>13s} {'select. end':>11s} {'w_bas chg':>9s}")
    for r in rows:
        print(f"{r['config']:40s} {r['n_failed']:6d} {r['expert_proxy']:12.0f} {r['final_perf']:10.3f} {r['task_frac_end']:13.3f} "
              f"{r['selectivity_end']:11.3f} {r['w_bas_change']:9.2e}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, help='sweep name (subfolder of sandbox/results)')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='override a default hyperparameter')
    ap.add_argument('--grid', default='{}', help='JSON dict {hp: [values]} or path to such a file')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    base = {k: parse_value(v) for k, v in (kv.split('=', 1) for kv in a.set)}
    grid = json.load(open(a.grid)) if Path(a.grid).is_file() else json.loads(a.grid)
    unknown = set(base) | set(grid)
    unknown -= set(DEFAULT_HP) | {'oja_decay', 'gate_power'}
    if unknown:
        print(f"warning: hyperparameters not in DEFAULT_HP (rule-specific?): {sorted(unknown)}")
    run_sweep(a.name, base, grid, a.seeds, a.workers)


if __name__ == '__main__':
    main()
