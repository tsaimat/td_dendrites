"""Compact ranked table of one or more sweeps, one column per varied hyperparameter:
    python -m sandbox.tab <sweep_name> [<sweep_name> ...]
Ranked by (seeds not reaching expert + diverged seeds), then median expert trial. Reads only summary.csv.
"""
import csv, re, sys
KEYS = ['rule', 'gh_post', 'gh_theta_scale', 'n_z', 'n_noise', 'init', 'init_norm', 'init_power', 'init_sigma', 'init_alpha', 'n_trials', 'lr_trace', 'reset_at_lift', 'w_bas_scale', 'bound', 'lr_bas', 'lr_ap', 'lr_policy', 'lr_td', 'gate_power', 'theta_0', 'theta_bas_0', 'theta_gate', 'lr_dep', 'dep_form', 'rule_sees_gain', 'oja_decay', 'burst_tau', 'burst_kappa', 'us_tau', 'bcm_tau', 'bcm_e0_scale', 'ca_ltd_frac', 'ca_theta_p', 'ca_beta', 'ca_gamma_d']
for name in sys.argv[1:]:
    rows = list(csv.DictReader(open(f'sandbox/results/{name}/summary.csv')))
    def g(r, k):
        try: return float(r[k])
        except: return float('nan')
    def m(k, c):
        s = re.search(rf'(?:^|_){k}-([^_]+(?:_rng)?)', c); return s.group(1) if s else ''
    present = [k for k in KEYS if any(m(k, r['config']) for r in rows)]
    out = []
    for r in rows:
        c = r['config']
        out.append((int(g(r,'expert_proxy_n_not_reached'))+int(g(r,'diverged')), g(r,'expert_proxy'), [m(k, c) for k in present],
                    int(g(r,'diverged')), int(g(r,'expert_proxy_n_not_reached')), g(r,'expert_after_lift'), g(r,'perf_inhibited_end'), g(r,'final_perf'), g(r,'det_tone'), g(r,'det_texture'), g(r,'det_distractor'), g(r,'task_frac_end'), g(r,'selectivity_end'), g(r,'w_bas_sum_mean')))
    out.sort(key=lambda t: (t[0], t[1] if t[1]==t[1] else 9e9))
    w = [max(len(k), max(len(t[2][i]) for t in out)) for i, k in enumerate(present)]
    print(f"== {name}\n" + ' '.join(f"{k:>{w[i]}s}" for i, k in enumerate(present)) + "  ndiv n_no expert ex_lift p_inh p_end dTon dTex dDis taskfr selec w_sum")
    for t in out:
        print(' '.join(f"{v:>{w[i]}s}" for i, v in enumerate(t[2])) + f"  {t[3]:4d} {t[4]:4d} {t[1]:6.0f} {t[5]:7.0f} {t[6]:5.3f} {t[7]:5.3f} {t[8]:4.0f} {t[9]:4.0f} {t[10]:4.0f} {t[11]:6.3f} {t[12]:5.3f} {t[13]:5.2f}")
