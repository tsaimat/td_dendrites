"""Table of a scan_gainless output: per config the Smith expert trials over the seeds (non-expert seeds counted at the
run length for the mean), the number of expert seeds, the mean final performance and weight metrics, and the paired
difference to a reference config (the same seeds).
Usage: python -m sandbox.tab_gainless <out.jsonl> [reference name] [sort: name|score]"""
import json, sys
from collections import defaultdict
import numpy as np


def load(path):
    rows = defaultdict(dict)
    for line in open(path):
        d = json.loads(line)
        rows[d['name']][d['seed']] = d
    return rows


def score(runs):
    """Mean expert trial over the seeds, a seed that is not expert counted at its run length."""
    return float(np.mean([d['expert'] if d['expert'] is not None else d['n_trials'] for d in runs.values()]))


if __name__ == '__main__':
    rows = load(sys.argv[1])
    ref = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in rows else None
    order = sorted(rows, key=lambda n: score(rows[n])) if (len(sys.argv) > 3 and sys.argv[3] == 'score') else list(rows)
    print(f"{'config':44s} {'n':>3s} {'expert':>5s} {'score':>7s} {'median':>7s} {'fin':>5s} {'div':>3s} {'recr':>5s} "
          f"{'dw':>5s} {'texdet':>6s} {'paired':>14s}")
    for name in order:
        runs = rows[name]
        ex = [d['expert'] for d in runs.values() if d['expert'] is not None]
        div = sum(d['diverged'] is not None for d in runs.values())
        paired = ''
        if ref and name != ref:
            common = [s for s in runs if s in rows[ref]]
            a = np.array([runs[s]['expert'] if runs[s]['expert'] is not None else runs[s]['n_trials'] for s in common])
            b = np.array([rows[ref][s]['expert'] if rows[ref][s]['expert'] is not None else rows[ref][s]['n_trials']
                          for s in common])
            paired = f"{np.mean(a - b):+7.0f} {np.sum(a < b):2d}/{len(common)}"
        print(f"{name:44s} {len(runs):3d} {len(ex):5d} {score(runs):7.0f} "
              f"{np.median(ex) if ex else float('nan'):7.0f} {np.mean([d['final_perf'] for d in runs.values()]):5.2f} "
              f"{div:3d} {np.mean([d['recruited'] for d in runs.values()]):5.2f} "
              f"{np.mean([d['w_change'] for d in runs.values()]):5.2f} "
              f"{np.mean([d['tex_detectors'] for d in runs.values()]):6.3f} {paired:>14s}")
