"""Causal ablation (not a candidate mechanism): run a gated rule with the basal update of the tone row set to zero,
to test whether the divergence of us / burst at higher basal rates comes from the tone synapses. Runs 5 seeds of each
configuration with and without the ablation in parallel and prints the summary. Usage (from the repo root):
    python -m sandbox.diag_tone_ablation
"""
import json, numpy as np, torch
from multiprocessing import Pool
from l5apical.helper import *
import sandbox.rules as R
from sandbox.simulate import simulate_seed_bu, summarize

BASE = dict(n_z=203, init_power=12, lr_policy=0.128, lr_td=0.032, lr_ap=0.064, bound='l1')
CONFIGS = [dict(rule='us', lr_bas=0.004, us_tau=5), dict(rule='us', lr_bas=0.008, us_tau=5),
           dict(rule='burst', lr_bas=0.004, burst_tau=5, burst_kappa=0.2), dict(rule='burst', lr_bas=0.008, burst_tau=5, burst_kappa=0.2)]


def run(args):
    hp, seed, ablate = args
    torch.set_num_threads(1)
    base = R.RULES[hp['rule']]
    def wrapped(*a):
        dw = base(*a)
        if dw is not None and ablate:
            dw[0] = 0.
        return dw
    R.RULES[hp['rule']] = wrapped
    s = summarize(simulate_seed_bu({**BASE, **hp}, seed=seed))
    return hp['rule'], hp['lr_bas'], ablate, seed, s


if __name__ == '__main__':
    jobs = [(hp, seed, ablate) for hp in CONFIGS for ablate in (False, True) for seed in range(5)]
    with Pool(8) as p:
        res = p.map(run, jobs)
    for hp in CONFIGS:
        for ablate in (False, True):
            rows = [r[4] for r in res if r[0] == hp['rule'] and r[1] == hp['lr_bas'] and r[2] == ablate]
            ex = np.array([r['expert_proxy'] for r in rows])
            print(f"{hp['rule']:6s} lr_bas {hp['lr_bas']:.3f} tone {'frozen' if ablate else 'plastic'}: diverged {sum(r['diverged'] for r in rows):.0f}/5, "
                  f"expert {np.nanmedian(ex):6.0f} in {np.sum(~np.isnan(ex))}/5, final perf {np.mean([r['final_perf'] for r in rows]):.3f}, "
                  f"detectors tone {np.mean([r['det_tone'] for r in rows]):.1f} texture {np.mean([r['det_texture'] for r in rows]):.1f} "
                  f"distractor {np.mean([r['det_distractor'] for r in rows]):.1f}, tone w max {np.mean([r['w_bas_max'] for r in rows]):.2f}")
