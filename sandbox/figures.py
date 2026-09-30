"""
Reproduce the manuscript panels for a sandbox configuration (e.g. the tuned control regime, or a bottom-up
plasticity rule on top of it) without Matlab: simulate N_SEEDS seeds with store_full=True (1800 trials, and the
apical-inhibition protocol with 4000 trials), run the Python port of the Smith et al. estimator for the performance
traces and expert / learning trials, store the results in the format of results/*.pickle, and call the panel
functions of l5apical.panels with their file resolver redirected to these pickles. Panels that need the theta_0
sweep of the mixed model (fs12b_theta0) are skipped.

Usage (from the repo root):
    python -m sandbox.figures build <name> '<hp json>'   # simulate + Smith analysis -> sandbox/results/figures/<name>/
    python -m sandbox.figures plot <name>                # all manuscript panels -> sandbox/results/figures/<name>/panels/
    python -m sandbox.figures both <name> '<hp json>'
The hp json holds the sandbox hyperparameters (rule, rates, ...); n_trials, n_trials_inhibited and sort_neurons are set here.
"""
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
from l5apical.helper import *
from sandbox.simulate import simulate_seed_bu, K_HP, K_W_BAS_H
from sandbox.smith import run_analysis

FIG_DIR = Path(__file__).parent / 'results' / 'figures'


def _run(args):
    hp, seed = args
    torch.set_num_threads(1)
    r = simulate_seed_bu(hp, seed=seed)
    t_learn, t_expert, pmid, p05 = run_analysis(np.nan_to_num(np.asarray(r[K_CORRECT_TRIALS], float), nan=0.))  # diverged seeds: NaN trials count as errors
    r[K_PERFORMANCE] = pmid
    # the main code stores the Matlab (1-based) trials minus 1; keep them as ints because the panels slice with them
    r[K_LEARNING_T] = int(t_learn - 1) if np.isfinite(t_learn) else -1
    r[K_EXPERT_T] = int(t_expert - 1) if np.isfinite(t_expert) else N_TRIALS - 1
    return seed, r


def build(name: str, hp: dict, seeds: int = N_SEEDS, workers: int = 4) -> None:
    out = FIG_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    for fname, extra in (('default.pickle', dict(n_trials=N_TRIALS, n_trials_inhibited=0)),
                         ('perturbed.pickle', dict(n_trials=N_TRIALS_AP_INH, n_trials_inhibited=N_TRIALS))):
        full = {**hp, **extra, 'store_full': True, 'sort_neurons': True, 'gain_at_preferred': True}
        with Pool(workers) as p:
            res = dict(p.map(_run, [(full, s) for s in range(seeds)]))
        results = [res[s] for s in range(seeds)]
        for r in results:  # panels bin by K_W_BAS; keep the initial weights there (final ones stay in the snapshot history)
            r['w_bas_final'] = r[K_W_BAS]
            r[K_W_BAS] = r[K_W_BAS_H][0]
        with open(out / fname, 'wb') as f:
            pickle.dump(results, f)
        ex = [r[K_EXPERT_T] for r in results]
        print(f"{name}/{fname}: expert trials {['%.0f' % e for e in ex]}, diverged {[r['diverged_at'] for r in results]}")
    json.dump(hp, open(out / 'hp.json', 'w'), indent=1)


def plot(name: str, skip: tuple = ('fs12b_theta0',)) -> None:
    import l5apical.panels as P
    out = FIG_DIR / name
    paths = {Simulation.DEFAULT: out / 'default.pickle', Simulation.MIXED_SELECTIVITY: out / 'default.pickle',
             Simulation.APICAL_INHIBITION: out / 'perturbed.pickle'}
    n_trials = {Simulation.DEFAULT: N_TRIALS, Simulation.MIXED_SELECTIVITY: N_TRIALS, Simulation.APICAL_INHIBITION: N_TRIALS_AP_INH}
    P.get_path = lambda simulation=Simulation.DEFAULT, theta_0=THETA_0: paths[simulation]
    P.get_params = lambda simulation=Simulation.DEFAULT, theta_0=THETA_0: (None,) * 8 + (n_trials[simulation],)
    P.PLOT_DIR = out / 'panels'
    P.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    (P.PLOT_DIR / P.SVG_DIR).mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    _save = P.save_or_show
    def save_png_too(saving, plot_dir, plot_name, plot_dpi=600):  # a PNG copy for quick viewing
        matplotlib.pyplot.savefig(plot_dir / f"{plot_name}.png", dpi=150)
        _save(saving, plot_dir, plot_name, plot_dpi)
    P.save_or_show = save_png_too
    names = [n for n in ('f4c_performance_eg', 'f4d_expert_v_pred', 'f4e_apical_dendrites_raster_plots', 'f4e_soma_raster_plots',
                         'f4f_sen_dendrite', 'f4g_apical_window_traces', 'f4h_performance', 'f4h_expert_trials', 'f4h_w_ap',
                         'fs11bc_transfer_f', 'fs11d_expert_v_pred', 'fs11e_apical_dendrites_raster_plots', 'fs12a_performance_eg',
                         'fs12b_wap_traces', 'fs12b_theta0', 'fs12c_selectivity_traces', 'fs12cd_selectivity_distribution',
                         'fs13_apical_dendrites_traces_boxplots') if n not in skip]
    for n in names:
        try:
            getattr(P, n)(saving=True)
            print(f'{n}: ok')
        except Exception as e:  # keep going, report at the end
            print(f'{n}: FAILED {type(e).__name__}: {e}')
        matplotlib.pyplot.close('all')


if __name__ == '__main__':
    cmd, name = sys.argv[1], sys.argv[2]
    if cmd in ('build', 'both'):
        build(name, json.loads(sys.argv[3]))
    if cmd in ('plot', 'both'):
        plot(name)
