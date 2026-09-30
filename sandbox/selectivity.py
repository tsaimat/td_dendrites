"""
Compare how the bottom-up plasticity rules change the selectivity of the pyramidal neurons relative to the control,
from the figure pickles built by sandbox.figures (default.pickle of each named configuration). Usage:
    python -m sandbox.selectivity control hebb bcm calcium
Writes sandbox/results/figures/compare_<names>.png/.pdf with one column per configuration:
  row 1  basal texture selectivity w_T1 - w_T2 per neuron, initial (grey) and final (colour) distributions
  row 2  each neuron's final vs initial strongest texture weight (go neurons black, no-go grey), with the tone
         weight (blue) and the strongest distractor weight (cyan) as extra scatters
  row 3  response selectivity x_som(go) - x_som(no-go) in the last 100 trials against the initial basal selectivity
  row 4  number of tone / texture / distractor detectors (strongest weight above 0.5) over trials
plus a summary printed to stdout (fractions of responsive, go-, no-go- and non-selective neurons per phase as in
fs12cd, and the expert trials).
"""
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from l5apical.helper import *
from l5apical.panels import get_texture_specific, COL_T1, COL_T2, COL_DISTRACTOR, COL_PRE, COL_OUTCOME
from sandbox.simulate import K_W_BAS_H, K_W_BAS_T, detector_counts
from sandbox.figures import FIG_DIR


def load(name):
    return pickle.load(open(FIG_DIR / name / 'default.pickle', 'rb'))


def phase_fractions(results, phase, tr_threshold=0.1, sel=0.2):
    """As in fs12cd: fractions of unresponsive, non-selective, go-selective and no-go-selective neurons in a phase."""
    n = 0; unresp = 0; nonsel = 0; go = 0; nogo = 0
    for r in results:
        x1 = get_texture_specific(True, r[K_OUTCOME], r[K_X_SOM_TXT])[phase[0]:phase[1]]
        x2 = get_texture_specific(False, r[K_OUTCOME], r[K_X_SOM_TXT])[phase[0]:phase[1]]
        resp = r[K_X_SOM_TXT][phase[0]:phase[1]].mean(0) > tr_threshold
        d = np.nanmean(x1, 0) - np.nanmean(x2, 0)
        n += len(resp); unresp += int((~resp).sum())
        nonsel += int((resp & (np.abs(d) < sel)).sum()); go += int((resp & (d >= sel)).sum()); nogo += int((resp & (d <= -sel)).sum())
    return np.array([unresp, nonsel, go, nogo]) / n


def compare(names):
    data = {n: load(n) for n in names}
    fig, axs = plt.subplots(4, len(names), figsize=(3.2 * len(names), 11), squeeze=False)
    for c, name in enumerate(names):
        res = data[name]
        w0 = np.stack([r[K_W_BAS_H][0] for r in res]); w1 = np.stack([r[K_W_BAS_H][-1] for r in res])  # seeds x stim x neurons
        # row 1: basal selectivity distributions
        ax = axs[0, c]
        bins = np.linspace(-1, 1, 41)
        ax.hist((w0[:, T1_IDX] - w0[:, T2_IDX]).ravel(), bins=bins, color='lightgrey', label='initial')
        ax.hist((w1[:, T1_IDX] - w1[:, T2_IDX]).ravel(), bins=bins, histtype='step', color='k', label='final')
        ax.set_yscale('log'); ax.set_xlabel('$w^{bas}_{Go} - w^{bas}_{NoGo}$'); ax.set_title(name)
        if c == 0: ax.set_ylabel('neurons'); ax.legend(frameon=False)
        # row 2: final vs initial weights
        ax = axs[1, c]
        go = w0[:, T1_IDX] >= w0[:, T2_IDX]
        wt0, wt1 = np.maximum(w0[:, T1_IDX], w0[:, T2_IDX]), np.maximum(w1[:, T1_IDX], w1[:, T2_IDX])
        ax.scatter(wt0[go], wt1[go], s=4, c=COL_T1, label='go texture', alpha=0.6)
        ax.scatter(wt0[~go], wt1[~go], s=4, c=COL_T2, label='no-go texture', alpha=0.6)
        ax.scatter(w0[:, 0].ravel(), w1[:, 0].ravel(), s=4, c=COL_PRE, label='tone', alpha=0.4)
        ax.scatter(w0[:, 3:].max(1).ravel(), w1[:, 3:].max(1).ravel(), s=4, c=COL_DISTRACTOR, label='strongest distractor', alpha=0.4)
        ax.plot([0, 1], [0, 1], 'k:', lw=0.7); ax.set_xlabel('initial weight'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if c == 0: ax.set_ylabel('final weight'); ax.legend(frameon=False, fontsize=7)
        # row 3: response selectivity late vs initial basal selectivity
        ax = axs[2, c]
        n_ts = 100
        for s, r in enumerate(res):
            x1 = get_texture_specific(True, r[K_OUTCOME], r[K_X_SOM_TXT])[-n_ts:]
            x2 = get_texture_specific(False, r[K_OUTCOME], r[K_X_SOM_TXT])[-n_ts:]
            d = np.nanmean(x1, 0) - np.nanmean(x2, 0)
            b = w0[s, T1_IDX] - w0[s, T2_IDX]
            ax.scatter(b, d, s=4, c=np.where(d > 0, COL_T1, COL_T2), alpha=0.6)
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
        ax.set_xlabel('initial $w^{bas}_{Go} - w^{bas}_{NoGo}$'); ax.set_xlim(-1, 1); ax.set_ylim(-11, 11)
        if c == 0: ax.set_ylabel('$x^{som}(Go) - x^{som}(NoGo)$, last 100 trials')
        # row 4: detectors over trials
        ax = axs[3, c]
        ts = res[0][K_W_BAS_T]
        det = np.array([[list(detector_counts(r[K_W_BAS_H][i]).values()) for i in range(len(ts))] for r in res]).mean(0)
        for k, (lab, col) in enumerate(zip(('tone', 'texture', 'distractor'), (COL_PRE, COL_T1, COL_DISTRACTOR))):
            ax.plot(ts, det[:, k], color=col, label=lab)
        ax.set_xlabel('trial'); ax.set_ylim(0, None)
        if c == 0: ax.set_ylabel('detectors (max weight > 0.5)'); ax.legend(frameon=False)
        # printed summary
        ex = [r[K_EXPERT_T] for r in res]
        f0, f1 = phase_fractions(res, (0, 100)), phase_fractions(res, (N_TRIALS - 100, N_TRIALS))
        print(f"{name:10s} expert {np.mean(ex):6.0f} +- {np.std(ex):4.0f} | first 100 trials: unresponsive {f0[0]:.2f} non-selective {f0[1]:.2f} "
              f"go {f0[2]:.3f} no-go {f0[3]:.3f} | last 100: unresponsive {f1[0]:.2f} non-selective {f1[1]:.2f} go {f1[2]:.3f} no-go {f1[3]:.3f} "
              f"| final detectors tone/tex/dis {det[-1, 0]:.1f}/{det[-1, 1]:.1f}/{det[-1, 2]:.1f}")
    fig.tight_layout()
    out = FIG_DIR / f"compare_{'_'.join(names)}"
    fig.savefig(str(out) + '.png', dpi=130); fig.savefig(str(out) + '.pdf')
    print('written', out)




def fs12cd_wide(names, tr_threshold=0.1, x_max=10., n_ts=100):
    """fs12cd of the manuscript for several configurations, with the histogram range widened to +-x_max so that the
    gained detectors of the plasticity rules (response differences up to the maximal gain) are counted, and the pie
    fractions (unresponsive / non-selective / go / no-go) computed over all neurons. Rows: configurations; columns:
    first and last n_ts trials."""
    from l5apical.panels import set_style
    set_style()
    fig, axs = plt.subplots(len(names), 2, figsize=(7, 2.2 * len(names)), squeeze=False)
    bins = np.arange(-x_max, x_max + 0.2, 0.2); nb2 = len(bins) // 2 - 1
    for r_i, name in enumerate(names):
        res = load(name)
        nn = len(res) * N_Z
        for c, phase in enumerate(((0, n_ts), (N_TRIALS - n_ts, N_TRIALS))):
            ax = axs[r_i, c]
            d, n_resp = [], 0
            for r in res:
                resp = r[K_X_SOM_TXT][phase[0]:phase[1]].mean(0) > tr_threshold
                x1 = get_texture_specific(True, r[K_OUTCOME], r[K_X_SOM_TXT])[phase[0]:phase[1]]
                x2 = get_texture_specific(False, r[K_OUTCOME], r[K_X_SOM_TXT])[phase[0]:phase[1]]
                dd = (np.nanmean(x1, 0) - np.nanmean(x2, 0))[resp]
                d.append(dd); n_resp += int(resp.sum())
            d = np.clip(np.concatenate(d), -x_max + 0.1, x_max - 0.1)
            counts, _, patches = ax.hist(d, bins=bins)
            for b, p in enumerate(patches):
                p.set_facecolor(COL_T2 if b < nb2 else COL_T1 if b > nb2 + 1 else COL_DISTRACTOR)
            numbers = [nn - n_resp, counts[nb2:nb2 + 2].sum(), counts[nb2 + 2:].sum(), counts[:nb2].sum()]
            ins = ax.inset_axes([0.0, 0.45, 0.5, 0.55])
            ins.pie(numbers, labels=[f'{100 * n / nn:.1f}%' for n in numbers], labeldistance=1.15,
                    colors=['white', COL_DISTRACTOR, COL_T1, COL_T2], wedgeprops={'linewidth': 0.5, 'edgecolor': 'k'}, textprops={'size': 6})
            ax.set_xlim(-x_max, x_max); ax.set_ylim(0, 0.03 * nn); ax.set_yticks([0.01 * k * nn for k in range(4)])
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f'{100 * x / nn:g}'))
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_xlabel('$x^{som}(go) - x^{som}(no-go)$' if r_i == len(names) - 1 else '')
            ax.set_title(f"{name}, {'first' if c == 0 else 'last'} {n_ts} trials", fontsize=7)
        axs[r_i, 0].set_ylabel('neurons [%]')
    fig.tight_layout()
    out = FIG_DIR / f"fs12cd_wide_{'_'.join(names)}"
    fig.savefig(str(out) + '.png', dpi=130); fig.savefig(str(out) + '.pdf')
    print('written', out)


if __name__ == '__main__':
    compare(sys.argv[1:])
    fs12cd_wide(sys.argv[1:])
