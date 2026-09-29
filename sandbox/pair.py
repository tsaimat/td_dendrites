"""Join a sweep without inhibition with its inhibition-protocol twin (same configuration names) and print, per
configuration, the expert trial, the expert trial after the lift, their ratio and the seeds lost in either:
    python -m sandbox.pair <sweep_without_inhibition> <sweep_with_inhibition> [--sort ratio|expert]
"""
import csv, re, sys
a, b = sys.argv[1], sys.argv[2]
key = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == '--sort' else 'expert'
def load(name):
    return {r['config']: r for r in csv.DictReader(open(f'sandbox/results/{name}/summary.csv'))}
def g(r, k):
    try: return float(r[k])
    except: return float('nan')
A, B = load(a), load(b)
rows = []
for c in A:
    if c not in B: continue
    ra, rb = A[c], B[c]
    ex, lift = g(ra, 'expert_proxy'), g(rb, 'expert_after_lift')
    lost = int(g(ra, 'expert_proxy_n_not_reached') + g(ra, 'diverged') + g(rb, 'expert_proxy_n_not_reached') + g(rb, 'diverged'))
    rows.append((lost, ex, lift, lift / ex if ex == ex and ex > 0 else float('nan'), g(rb, 'perf_inhibited_end'), c))
rows.sort(key=lambda t: (t[0], abs(t[3] - 1) if key == 'ratio' else t[1]))
print(f"{'lost':>4} {'expert':>6} {'lift':>6} {'ratio':>5} {'p_inh':>5}  config")
for lost, ex, lift, ratio, pinh, c in rows:
    print(f"{lost:4d} {ex:6.0f} {lift:6.0f} {ratio:5.2f} {pinh:5.2f}  {c}")
