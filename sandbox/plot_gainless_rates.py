"""Learning against the policy learning rate, with and without the apical gain in the policy's input (2026-11-02):
fraction of seeds expert within the run, trials to expert (mean, non-expert seeds counted at the run length) and the
final performance (last 200 trials), per condition, from the scan_gainless outputs in sandbox/results/gainless/.
Usage: python -m sandbox.plot_gainless_rates <out.png> [seeds, e.g. 0-4]"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR = Path('sandbox/results/gainless')
CONDITIONS = {  # label: (config name pattern with the policy rate as group 1, color)
    'Gain (control4)': (r'ctrl(?:_p([\d.]+))?$', 'k'),
    'No gain (= apical inhibition)': (r'blind_p([\d.]+)$', 'grey'),
    'No gain + gated Hebb, live apical signal': (r'e1_gatedhebb_gp0\.25(?:_p([\d.]+))?$', '#2a78d6'),
    'No gain + gated Hebb, replayed control4 apical signal': (r'e2r_gatedhebb_gp0\.25(?:_p([\d.]+))?$', '#eb6834'),
}
FAST_LABEL = 'Gain (control3 rates, S15e sweep)'
FAST_COLOR = '#1baf7a'
DEFAULT_RATE = {'Gain (control4)': 0.004, 'No gain + gated Hebb, live apical signal': 0.256,
                'No gain + gated Hebb, replayed control4 apical signal': 0.256}


def collect(seeds):
    runs = defaultdict(dict)  # (label, rate) -> seed -> run
    for path in sorted(DIR.glob('s*.jsonl')):
        for line in open(path):
            d = json.loads(line)
            if d['seed'] not in seeds:
                continue
            for label, (pattern, _) in CONDITIONS.items():
                m = re.match(pattern, d['name'])
                if m:
                    rate = float(m.group(1)) if m.group(1) else DEFAULT_RATE[label]
                    runs[(label, rate)][d['seed']] = d
    return runs


def collect_fast(runs, seeds):
    """The main code's policy-rate sweep of the fast regime (the gained network at control3's other rates)."""
    from l5apical.helper import HYPERPARAMETERS, Simulation, get_learning_rates, K_EXPERT_T, K_CORRECT_TRIALS, N_TRIALS
    from l5apical.panels import get_results
    hp = HYPERPARAMETERS['fast']
    for value in sorted(get_learning_rates('lr_policy') + [hp['lr_policy']]):
        if value == hp['lr_policy']:
            results = get_results(simulation=Simulation.DEFAULT, theta_0=hp['theta_0'], fast=True)
        else:
            results = get_results(simulation=Simulation.DEFAULT, lr=('lr_policy', value), fast=True)
        for seed in seeds:
            r = results[seed]
            c = np.nan_to_num(np.asarray(r[K_CORRECT_TRIALS], float), nan=0.)
            runs[(FAST_LABEL, value)][seed] = dict(expert=None if np.isnan(r[K_EXPERT_T]) else int(r[K_EXPERT_T]),
                                                  n_trials=N_TRIALS, final_perf=float(c[-200:].mean()))


if __name__ == '__main__':
    out = sys.argv[1]
    a, b = (sys.argv[2] if len(sys.argv) > 2 else '0-4').split('-')
    seeds = set(range(int(a), int(b) + 1))
    runs = collect(seeds)
    collect_fast(runs, seeds)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for label, (_, color) in list(CONDITIONS.items()) + [(FAST_LABEL, (None, FAST_COLOR))]:
        rates = sorted(r for (lab, r) in runs if lab == label)
        stats = []
        for r in rates:
            rs = list(runs[(label, r)].values())
            expert = [d['expert'] for d in rs]
            stats.append((np.mean([e is not None for e in expert]),
                          np.mean([e if e is not None else d['n_trials'] for e, d in zip(expert, rs)]),
                          np.mean([d['final_perf'] for d in rs]), len(rs)))
            print(f'{label:52s} {r:6g}: {sum(e is not None for e in expert)}/{len(rs)} expert, '
                  f'mean {stats[-1][1]:.0f}, final {stats[-1][2]:.2f}, experts {sorted(e for e in expert if e is not None)}')
        if not rates:
            continue
        s = np.array(stats)
        for ax, k in zip(axs, range(3)):
            ax.plot(rates, s[:, k], 'o-', color=color, label=label, ms=4)
    for ax, ylabel in zip(axs, ('Fraction of seeds expert', 'Trials to expert (not expert: 1800)',
                                'Fraction correct, last 200 trials')):
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Policy learning rate')
        ax.set_ylabel(ylabel)
        ax.spines[['top', 'right']].set_visible(False)
    axs[2].axhline(0.5, color='k', ls=':', lw=0.8)
    fig.suptitle(f'Seeds {a}-{b}, 1800 trials', fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc='lower center', ncol=3)
    fig.savefig(out, dpi=200)
