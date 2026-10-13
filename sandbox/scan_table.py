"""Aggregate scan.jsonl files: per config, Smith expert fresh / after lift (mean +- sd), failures, recruitment, score."""
import json, sys, numpy as np
from collections import defaultdict
rows = defaultdict(lambda: defaultdict(list))
seed_filter = None
args = [a for a in sys.argv[1:] if not a.startswith('--seeds=')]
for a in sys.argv[1:]:
    if a.startswith('--seeds='):
        lo, hi = a[8:].split('-'); seed_filter = set(range(int(lo), int(hi) + 1))
for p in args:
    for l in open(p):
        d = json.loads(l)
        if seed_filter is None or d['seed'] in seed_filter:
            rows[d['name']][d['prot']].append(d)
out = []
for name, pr in rows.items():
    def stat(lst):
        e = np.array([d['expert'] if d['expert'] is not None else np.nan for d in lst], float)
        fail = sum(d['expert'] is None or d['diverged'] is not None for d in lst)
        return e, fail
    ef, ff = stat(pr.get('fresh', [])); ei, fi = stat(pr.get('inh', []))
    T = np.mean([d['trans'] for d in pr.get('fresh', []) if 'trans' in d], 0) * 100 if pr.get('fresh') else np.full((4, 4), np.nan)
    fg = np.mean([d['full_gain'] for d in pr.get('fresh', []) if 'full_gain' in d]) * 100 if pr.get('fresh') else np.nan
    mf, mi = np.nanmean(ef) if len(ef) else np.nan, np.nanmean(ei) if len(ei) else np.nan
    score = abs(mf - 1100) + abs(mi - 1100) + abs(mf - mi) + 300 * (ff + fi)
    out.append((score, f"{name:34s} n={len(ef):2d}/{len(ei):2d} fresh {mf:5.0f} +- {np.nanstd(ef):4.0f} | lift {mi:5.0f} +- {np.nanstd(ei):4.0f} | "
                       f"fail {ff}/{fi} | nonsel->sel {T[1, 2:].sum():4.1f} unresp->sel {T[0, 2:].sum():4.1f} sel {T[:, 2:].sum():4.1f} full gain {fg:4.1f} | score {score:5.0f}"))
for s, line in sorted(out, key=lambda x: (np.nan_to_num(x[0], nan=1e9))):
    print(line)
