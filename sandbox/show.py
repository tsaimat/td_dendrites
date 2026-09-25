"""Print the summary table of one or more sweeps:  python -m sandbox.show [--recompute] [--rank] <sweep_name> [<sweep_name> ...]
--rank sorts configurations best first: fewest seeds that never reached expert or diverged, then earliest expert trial."""
import csv
import sys
import numpy as np
from sandbox.sweep import RESULTS_DIR, print_summary, write_summary

args = [a for a in sys.argv[1:] if not a.startswith('--')]
for name in args:
    if '--recompute' in sys.argv:
        write_summary(RESULTS_DIR / name)
    rows = list(csv.DictReader(open(RESULTS_DIR / name / 'summary.csv')))
    for r in rows:
        for k, v in r.items():
            if k not in ('config', 'rule', 'bound', 'init'):
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
        r['n_failed'] = int(r['n_failed']); r['expert_proxy_n_not_reached'] = int(r['expert_proxy_n_not_reached'])
        r['diverged'] = int(r['diverged']) if 'diverged' in r else 0
    if '--rank' in sys.argv:
        rows.sort(key=lambda r: (r['expert_proxy_n_not_reached'] + r['diverged'],
                                 r['expert_proxy'] if not np.isnan(r['expert_proxy']) else 1e9))
    print(f"== {name}")
    print_summary(rows)
