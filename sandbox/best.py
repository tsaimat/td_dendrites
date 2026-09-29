"""Print the best configurations of a sweep per rule and init power (reads only summary.csv):
    python -m sandbox.best <sweep_name> [n_per_group]
"""
import csv, re, sys
name = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
rows = list(csv.DictReader(open(f'sandbox/results/{name}/summary.csv')))
def g(r, k):
    try: return float(r[k])
    except: return float('nan')
groups = {}
for r in rows:
    m = re.search(r'rule-(\w+?)_', r['config']); p = re.search(r'init_power-(\d+)', r['config'])
    groups.setdefault((p.group(1) if p else '', m.group(1) if m else ''), []).append(r)
for key in sorted(groups):
    sel = sorted(groups[key], key=lambda r: (g(r, 'expert_proxy_n_not_reached') + g(r, 'diverged'), g(r, 'expert_proxy')))
    print(f'--- power {key[0]} {key[1]} ({len(sel)} configs; fail/div, expert, lift, p_inh, final perf, det tone/tex/dis, w_sum)')
    for r in sel[:n]:
        c = re.sub(r'rule-\w+?_bound-\w+?_|_?init_power-\d+_?', '', r['config'])
        print(f"  {int(g(r,'expert_proxy_n_not_reached'))}/{int(g(r,'diverged'))}  {g(r,'expert_proxy'):5.0f} {g(r,'expert_after_lift'):5.0f} {g(r,'perf_inhibited_end'):.2f}  {g(r,'final_perf'):.3f}  {g(r,'det_tone'):3.0f}/{g(r,'det_texture'):3.0f}/{g(r,'det_distractor'):3.0f}  {g(r,'w_bas_sum_mean'):.2f}  {c}")
