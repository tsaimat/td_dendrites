"""
Summary figure of the speed-up by bottom-up plasticity in one or more control conditions, from the figure pickles
built by sandbox.figures (Smith performance traces and expert trials). Usage:
    python -m sandbox.speedup slow:control2,hebb2,bcm2,calcium2 fast:control3,hebb3,bcm3,calcium3
Each argument is '<condition label>:<comma-separated figure-set names>'; the first set of each condition is its
control. Writes sandbox/results/figures/speedup_<labels>.png/.pdf with one column per condition:
  row 1  mean Smith performance trace over seeds without apical inhibition
  row 2  the same in the apical-inhibition protocol (inhibited for the first N_TRIALS trials)
  row 3  per-seed Smith expert trials (dots), mean +- sd (bars), without inhibition and after the lift
and prints the means, the ratios to the control and the post-lift to no-inhibition ratios.
"""
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from l5apical.helper import *
from sandbox.figures import FIG_DIR

COLORS = ['#7f7f7f', '#2985C4', '#ce583a', '#77c044', '#9f5fb0', '#eda922']


def load(name):
    d = pickle.load(open(FIG_DIR / name / 'default.pickle', 'rb'))
    p = pickle.load(open(FIG_DIR / name / 'perturbed.pickle', 'rb'))
    return dict(perf=np.stack([r[K_PERFORMANCE][:N_TRIALS] for r in d]),
                perf_inh=np.stack([r[K_PERFORMANCE][:N_TRIALS_AP_INH] for r in p]),
                expert=np.array([r[K_EXPERT_T] for r in d], float),
                lift=np.array([r[K_EXPERT_T] - N_TRIALS for r in p], float),
                diverged=[r.get('diverged_at') for r in d] + [r.get('diverged_at') for r in p],
                label=name.rstrip('0123456789'))


def speedup(conditions: dict) -> None:
    n = len(conditions)
    fig, axs = plt.subplots(3, n, figsize=(4.2 * n, 9.5), squeeze=False)
    for c, (cond, names) in enumerate(conditions.items()):
        data = [load(nm) for nm in names]
        for row, key, t_max in ((0, 'perf', N_TRIALS), (1, 'perf_inh', N_TRIALS_AP_INH)):
            ax = axs[row, c]
            ax.axhline(0.5, color='k', ls=':', lw=0.8)
            if row == 1:
                ax.axvspan(0, N_TRIALS, color='0.93', zorder=0)
            for i, d in enumerate(data):
                m, s = d[key].mean(0), d[key].std(0)
                ax.plot(np.arange(t_max), m, color=COLORS[i], label=d['label'])
                ax.fill_between(np.arange(t_max), m - s, m + s, color=COLORS[i], alpha=0.15, lw=0)
            ax.set_ylim(0.4, 1); ax.set_xlim(0, t_max); ax.set_xlabel('Trial')
            if c == 0:
                ax.set_ylabel('Performance' + ('' if row == 0 else ' (apical inhibition protocol)'))
            if row == 0:
                ax.set_title(f"{cond}: control {data[0]['expert'].mean():.0f} trials")
                ax.legend(frameon=False, loc='lower right')
        ax = axs[2, c]
        print(f"== {cond}")
        for i, d in enumerate(data):
            for j, key in enumerate(('expert', 'lift')):
                x = i + (j - 0.5) * 0.3
                v = d[key]
                ax.bar(x, v.mean(), width=0.26, color=COLORS[i], alpha=0.35 if j else 0.7, edgecolor='none')
                ax.errorbar(x, v.mean(), v.std(), color='k', lw=1, capsize=2)
                ax.scatter(x + np.random.default_rng(0).uniform(-0.08, 0.08, len(v)), v, s=8, color='k', zorder=3)
            ex, li, ctrl = d['expert'], d['lift'], data[0]
            print(f"  {d['label']:8s} expert {ex.mean():5.0f} +- {ex.std():3.0f} ({ex.min():.0f} to {ex.max():.0f}; "
                  f"{ex.mean() / ctrl['expert'].mean():.2f} of control)   after lift {li.mean():5.0f} +- {li.std():3.0f} "
                  f"({li.mean() / ctrl['lift'].mean():.2f} of control)   lift / no-inhibition {li.mean() / ex.mean():.2f}"
                  f"   diverged {sum(x is not None for x in d['diverged'])} of {len(d['diverged'])}")
        ax.set_xticks(range(len(data))); ax.set_xticklabels([d['label'] for d in data])
        ax.set_ylim(0, 1.25 * max(np.max(d[k]) for d in data for k in ('expert', 'lift')))
        if c == 0:
            ax.set_ylabel('Expert trial')
        ax.text(0.98, 0.97, 'dark: no inhibition\nlight: after the lift', transform=ax.transAxes, va='top', ha='right', fontsize=8)
    fig.tight_layout()
    stem = FIG_DIR / ('speedup_' + '_'.join(conditions))
    fig.savefig(f'{stem}.png', dpi=150); fig.savefig(f'{stem}.pdf')
    print(f'-> {stem}.png')


if __name__ == '__main__':
    speedup({a.split(':')[0]: a.split(':')[1].split(',') for a in sys.argv[1:]})
