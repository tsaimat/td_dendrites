import json, sys, numpy as np
from collections import defaultdict
R = defaultdict(lambda: defaultdict(list))
for p in sys.argv[1:]:
    for l in open(p):
        d = json.loads(l); R[d['name']][d['prot']].append(d)
rows = []
for n, P in R.items():
    def st(L):
        e = np.array([d['expert'] if d['expert'] is not None else np.nan for d in L], float)
        return e, sum(d['diverged'] is not None for d in L), sum(d['expert'] is None for d in L)
    ef, df, ff = st(P.get('fresh', [])); ei, di, fi = st(P.get('inh', []))
    L = P.get('fresh', [])
    T = np.mean([d['trans'] for d in L if 'trans' in d], 0) * 100 if any('trans' in d for d in L) else np.full((4, 4), np.nan)
    mi = np.nanmean(ei) if len(ei) else np.nan
    rows.append((n.split('_')[0], (df + ff + di + fi) * 1e4 + np.nanmean(ef), f"{n:30s} fresh {np.nanmean(ef):5.0f} +- {np.nanstd(ef):4.0f} div/fail {df}/{ff} | lift {mi:5.0f} +- {np.nanstd(ei) if len(ei) else np.nan:4.0f} div/fail {di}/{fi} | unresp end {T[:,0].sum():4.1f} sel end {T[:,2:].sum():4.1f} nonsel->sel {T[1,2:].sum():4.1f}"))
for r in sorted(rows): print(r[2])
