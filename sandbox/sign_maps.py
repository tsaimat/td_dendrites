"""
Instantaneous basal weight update of every rule over the (apical gain, basal drive) plane of this model, with the
slow variables frozen at their baseline (burst: Pbar at the resting apical activation; bcm: theta_M = theta).
Adapted from the demo at the bottom of sandbox/plasticity_rules_tmp.py to the model's variables.

    python -m sandbox.sign_maps            # writes sandbox/results/sign_maps.png

The plotted quantity is dw / lr_bas for one active synapse (x_in = 1) on a neuron with basal drive x_bas (the weight
of the active synapse plus background) and apical activation set by the gain on the x axis; the weight bound is not
applied. theta is the median per-neuron threshold of the default random initialization.
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from l5apical.helper import gain, apical_transfer, MAX_GAIN, BKG
from sandbox.rules import RULES, A_REST
from sandbox.simulate import DEFAULT_HP, get_params_bu

OUT = Path(__file__).parent / 'results' / 'sign_maps.png'
RULE_TITLES = {
    'hebb': 'hebb (main code)\n$x_{som} - \\theta$',
    'burst': 'burst (Payeur 2021)\n$x_{som}\\,(P - \\bar P)$',
    'bcm': 'bcm (sliding threshold)\n$x_{som}\\,(x_{som} - \\theta_M)$',
    'us': 'Urbanczik-Senn 2014\n$x_{bas}\\,(g - 1)$',
    'calcium': 'calcium thresholds\n(Shouval 2002, Graupner & Brunel 2012)',
}


def main(rules=tuple(RULE_TITLES), n=200):
    torch.manual_seed(0)
    hp = {**DEFAULT_HP, 'burst_tau': float('inf'), 'bcm_tau': float('inf'), '_steps_per_trial': 1}
    _, _, _, _, _, w_bas, thetas, _, _ = get_params_bu(hp)
    theta = float(thetas.median())

    g = torch.linspace(1, MAX_GAIN, n)                       # apical gain on the x axis
    x_ap = A_REST + (g - 1) / (MAX_GAIN - 1) * (1 - A_REST)  # the apical activation that produces this gain
    w = torch.linspace(0, 1, n)                              # weight of the active synapse on the y axis
    G, W = torch.meshgrid(g, w, indexing='xy')
    XA = torch.meshgrid(x_ap, w, indexing='xy')[0]
    x_bas = (W + BKG).ravel()
    gains = gain(XA.ravel())
    x_som = gains * x_bas
    m = x_bas.numel()
    th = torch.full((m,), theta)
    w_max = float((w_bas.max(0).values + BKG).median())  # median strongest initial weight (calcium threshold ref)
    state = {'p_bar': torch.full((m,), A_REST), 'theta_m': th.clone(), 'e0': torch.ones(m),
             '_e_x_som_sq_init': torch.ones(m), '_w_max_init': torch.full((m,), w_max)}

    fig, axes = plt.subplots(1, len(rules), figsize=(3.3 * len(rules), 3.6), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), rules):
        dw = RULES[name](None, torch.ones(1), x_som, x_bas, XA.ravel(), gains, th, hp, state)
        Z = (dw[0] / hp['lr_bas']).reshape(n, n).numpy()
        vmin, vmax = min(Z.min(), -1e-9), max(Z.max(), 1e-9)
        pc = ax.pcolormesh(G.numpy(), W.numpy(), Z, cmap='RdBu_r', norm=TwoSlopeNorm(0, vmin=vmin, vmax=vmax),
                           rasterized=True)
        if Z.min() < 0 < Z.max():
            ax.contour(G.numpy(), W.numpy(), Z, levels=[0], colors='k', linewidths=0.8, linestyles='--')
        ax.axhline(theta - BKG, color='k', lw=0.6, ls=':')
        if name == 'calcium':
            ax.axhline(w_max - BKG, color='k', lw=0.6, ls='-.')
        ax.set_title(RULE_TITLES.get(name, name), fontsize=8.5)
        ax.set_xlabel('apical gain $g$ (1 = apical at rest)')
        cb = fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label('$\\mathrm{d}w / \\mathrm{lr}$', fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel('weight of the active basal synapse $w$')
    fig.suptitle(f'basal update of one active synapse; red LTP, blue LTD, dashed dw = 0, dotted w = theta - BKG '
                 f'(theta = {theta:.3f}, median of the random init), dash-dotted: median strongest initial weight '
                 f'({w_max - BKG:.2f})', fontsize=9, y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
