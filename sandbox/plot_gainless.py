"""Performance (fraction correct per 200-trial block, mean +- sd over the seeds) and lick rate of scan_gainless configs.
Usage: python -m sandbox.plot_gainless <out.png> <jsonl>:<config>:<label> [...]"""
import json, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['k', '#1baf7a', 'grey', '#2a78d6', '#eb6834', '#d6a62a', '#4a3aa7']


def load(path, name):
    return [d for d in map(json.loads, open(path)) if d["name"] == name and d["seed"] < 5]  # seeds 0-4


if __name__ == '__main__':
    out, specs = sys.argv[1], [s.split(':', 2) for s in sys.argv[2:]]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, (path, name, label) in enumerate(specs):
        runs = load(path, name)
        if not runs:
            print('no runs for', name)
            continue
        n_expert = sum(r['expert'] is not None for r in runs)
        for ax, key in zip(axs, ('perf_200', 'lick_200')):
            if key not in runs[0]:
                continue
            v = np.array([r[key] for r in runs])
            x = 100 + 200 * np.arange(v.shape[1])
            ax.plot(x, v.mean(0), color=COLORS[k % len(COLORS)],
                    label=f'{label} ({n_expert}/{len(runs)} expert)' if key == 'perf_200' else None)
            ax.fill_between(x, v.mean(0) - v.std(0), v.mean(0) + v.std(0), color=COLORS[k % len(COLORS)], alpha=0.07,
                            lw=0)
    axs[0].axhline(0.5, color='k', ls=':', lw=0.8)
    axs[0].set(xlabel='Trial', ylabel='Fraction correct (200-trial blocks)', ylim=(0.4, 1))
    axs[1].set(xlabel='Trial', ylabel='Lick rate (200-trial blocks)', ylim=(0, 1.02))
    axs[0].legend(frameon=False, fontsize=8, loc='upper left')
    for ax in axs:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
