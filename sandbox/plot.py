"""
Plot the outcome of a sweep produced by sandbox.sweep.
    python -m sandbox.plot <sweep_name>
Writes <sweep>/<config>.png (one overview per configuration) and <sweep>/comparison.png.
"""
import sys
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from l5apical.helper import K_CORRECT_TRIALS, K_W_AP, T1_IDX, T2_IDX
from sandbox.simulate import (K_W_BAS_H, K_W_BAS_T, K_HP, K_STATE_H, moving_performance, expert_trial_proxy,
                              task_weight_fraction, texture_selectivity, detector_counts)

RESULTS_DIR = Path(__file__).parent / 'results'


def plot_config(name: str, results: list[dict], out: Path) -> None:
    fig, axs = plt.subplots(2, 4, figsize=(17, 7))
    fig.suptitle(name)
    n_seeds = len(results)
    cols = plt.cm.viridis(np.linspace(0, 0.9, n_seeds))
    n_inh = int(results[0][K_HP]['n_trials_inhibited'])

    ax = axs[0, 0]
    for s, r in enumerate(results):
        ax.plot(moving_performance(r[K_CORRECT_TRIALS]), color=cols[s], lw=0.8)
        e = expert_trial_proxy(r[K_CORRECT_TRIALS])
        if not np.isnan(e):
            ax.axvline(e, color=cols[s], ls=':', lw=0.8)
    ax.axhline(0.8, color='k', ls='--', lw=0.5)
    if n_inh > 0:
        ax.axvspan(0, n_inh, color='0.9', zorder=0)
    ax.set(xlabel='trial', ylabel='moving performance (100 trials)', title='performance, dotted = expert proxy')

    ax = axs[0, 1]
    for s, r in enumerate(results):
        w = r[K_W_AP]
        ax.plot(w.mean(1), color=cols[s], lw=0.8, label='mean' if s == 0 else None)
        ax.plot(w.max(1), color=cols[s], lw=0.8, ls='--', label='max' if s == 0 else None)
    ax.set(xlabel='trial', ylabel=r'$w^{ap}$', title='apical weights over neurons'); ax.legend()

    ax = axs[0, 2]
    for s, r in enumerate(results):
        t, wh = r[K_W_BAS_T], r[K_W_BAS_H]
        ax.plot(t, [np.abs(w - wh[0]).mean() for w in wh], color=cols[s], lw=0.8)
    ax.set(xlabel='trial', ylabel=r'mean $|w^{bas} - w^{bas}_0|$', title='basal weight drift')

    ax = axs[1, 0]
    for s, r in enumerate(results):
        t, wh = r[K_W_BAS_T], r[K_W_BAS_H]
        ax.plot(t, [task_weight_fraction(w).mean() for w in wh], color=cols[s], lw=0.8)
    ax.set(xlabel='trial', ylabel='task-stimulus weight fraction', title='mean over neurons')

    ax = axs[1, 1]
    styles = {'det_texture': '-', 'det_tone': ':', 'det_distractor': '--'}
    for s, r in enumerate(results):
        t, wh = r[K_W_BAS_T], r[K_W_BAS_H]
        counts = [detector_counts(w) for w in wh]
        for k, ls in styles.items():
            ax.plot(t, [c[k] for c in counts], color=cols[s], lw=0.8, ls=ls, label=k[4:] if s == 0 else None)
    ax.set(xlabel='trial', ylabel='neurons with max weight > 0.5', title='detectors by dominant input'); ax.legend()

    ax = axs[1, 2]
    w0 = np.concatenate([r[K_W_BAS_H][0].ravel() for r in results])
    w1 = np.concatenate([r[K_W_BAS_H][-1].ravel() for r in results])
    bins = np.linspace(0, 1, 41)
    ax.hist(w0, bins, alpha=0.5, label='start', log=True)
    ax.hist(w1, bins, alpha=0.5, label='end', log=True)
    ax.set(xlabel=r'$w^{bas}$', ylabel='count', title='basal weight distribution'); ax.legend()

    ax = axs[0, 3]
    for s, r in enumerate(results):
        t, wh = r[K_W_BAS_T], r[K_W_BAS_H]
        ax.plot(t, [w.sum(0).mean() for w in wh], color=cols[s], lw=0.8, label='mean' if s == 0 else None)
        ax.plot(t, [w.sum(0).max() for w in wh], color=cols[s], lw=0.8, ls='--', label='max' if s == 0 else None)
    ax.set(xlabel='trial', ylabel=r'$\sum_i w^{bas}_i$', title='total basal weight per neuron'); ax.legend()

    ax = axs[1, 3]
    styles = ['-', '--', ':']
    for s, r in enumerate(results):
        for i, (k, v) in enumerate(sorted(r[K_STATE_H].items())):
            ax.plot(v, color=cols[s], lw=0.8, ls=styles[i % 3], label=k if s == 0 else None)
    ax.set(xlabel='trial', title='rule slow variables, mean over neurons')
    if results[0][K_STATE_H]:
        ax.legend()
    for a in axs.ravel():
        if n_inh > 0 and a is not axs[0, 0] and a is not axs[1, 2]:
            a.axvspan(0, n_inh, color='0.9', zorder=0)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_comparison(sweep: dict[str, list[dict]], out: Path) -> None:
    names = list(sweep)
    fig, axs = plt.subplots(1, 3, figsize=(max(9, 1.2 * len(names) + 4), 4))
    def expert(r):
        n_inh = int(r[K_HP]['n_trials_inhibited'])
        return expert_trial_proxy(r[K_CORRECT_TRIALS][n_inh:])
    metrics = [('expert trial (proxy, after lift if inhibited)', expert),
               ('final task weight fraction', lambda r: task_weight_fraction(r[K_W_BAS_H][-1]).mean()),
               ('final texture selectivity', lambda r: texture_selectivity(r[K_W_BAS_H][-1]).mean())]
    for ax, (label, fn) in zip(axs, metrics):
        for i, n in enumerate(names):
            vals = np.array([fn(r) for r in sweep[n]], dtype=float)
            ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.1, 0.1, len(vals)), vals, s=12, alpha=0.7)
            if np.any(~np.isnan(vals)):
                ax.plot([i - 0.3, i + 0.3], [np.nanmean(vals)] * 2, 'k', lw=1.5)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(name: str) -> None:
    d = RESULTS_DIR / name
    sweep = {}
    for p in sorted(d.glob('*.pickle')):
        with open(p, 'rb') as f:
            runs = [r for r in pickle.load(f) if not isinstance(r, str)]  # drop failed seeds
        if not runs:
            print(f"  {p.stem}: all seeds failed, skipped")
            continue
        sweep[p.stem] = runs
        plot_config(p.stem, runs, d / f"{p.stem}.png")
    if sweep:
        plot_comparison(sweep, d / 'comparison.png')
    print(f"wrote {len(sweep)} config figures and comparison.png to {d}")


if __name__ == '__main__':
    main(sys.argv[1])
