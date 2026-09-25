"""
Overlay the performance curves and detector counts of selected configurations, possibly from different sweeps.

    python -m sandbox.compare <out.png> <sweep>/<config> [<sweep>/<config> ...]

Thin lines are seeds, thick lines the mean over seeds. A grey band marks the apical inhibition period.
"""
import sys
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from l5apical.helper import K_CORRECT_TRIALS
from sandbox.simulate import K_W_BAS_H, K_W_BAS_T, K_HP, moving_performance, detector_counts

RESULTS_DIR = Path(__file__).parent / 'results'


def main(out: str, specs: list[str]) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    cols = plt.cm.tab10(np.arange(len(specs)))
    for c, spec in zip(cols, specs):
        sweep, config = spec.split('/', 1)
        with open(RESULTS_DIR / sweep / f"{config}.pickle", 'rb') as f:
            runs = [r for r in pickle.load(f) if not isinstance(r, str)]
        n_inh = int(runs[0][K_HP]['n_trials_inhibited'])
        perfs = np.array([moving_performance(r[K_CORRECT_TRIALS]) for r in runs])
        for p in perfs:
            axs[0].plot(p, color=c, lw=0.5, alpha=0.4)
        axs[0].plot(np.nanmean(perfs, 0), color=c, lw=2, label=spec)
        for i, key in enumerate(['det_texture', 'det_distractor']):
            t = runs[0][K_W_BAS_T]
            counts = np.array([[detector_counts(w)[key] for w in r[K_W_BAS_H]] for r in runs], dtype=float)
            for k in counts:
                axs[1 + i].plot(t, k, color=c, lw=0.5, alpha=0.4)
            axs[1 + i].plot(t, np.nanmean(counts, 0), color=c, lw=2)
        if n_inh > 0:
            for ax in axs:
                ax.axvspan(0, n_inh, color='0.92', zorder=0)
    axs[0].axhline(0.8, color='k', ls='--', lw=0.6)
    axs[0].set(xlabel='trial', ylabel='moving performance (100 trials)', title='performance')
    axs[0].legend(fontsize=7)
    axs[1].set(xlabel='trial', ylabel='neurons with max weight > 0.5', title='texture detectors')
    axs[2].set(xlabel='trial', ylabel='neurons with max weight > 0.5', title='distractor detectors')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / out, dpi=120)
    print(f"wrote {RESULTS_DIR / out}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
