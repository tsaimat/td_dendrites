"""
Bottom-up (basal) plasticity rules written in the user's model variables
========================================================================

Architecture (apical gain modulation):

    x_pre   presynaptic firing rates (vector, Hz)
    w_bas   basal synaptic weights   (vector, plastic)
    x_bas   = sum_i w_bas_i * x_pre_i          basal drive
    x_ap    apical activity (a.u.)
    x_som   = gain(x_ap) * x_bas               somatic firing rate (Hz)

The multiplicative apical gain is the Larkum, Senn & Luscher (Cereb
Cortex 2004) result: distal tuft input increases the slope of the
somatic f-I curve via bAP-activated dendritic Ca2+ conductances, and the
gain increase is accompanied by a switch from isolated spikes to
BURSTING. That last point is what licenses reading the burst fraction
P off x_ap (and x_som) in rule 1 below.

Four rules, each exposing  .step(x_pre, x_bas, x_som, x_ap, dt) -> dw_bas
(so they drop into an existing simulation loop; internal slow state is
updated inside step()).

1. BurstRule    -- Payeur et al., Nat Neurosci 2021 (RECOMMENDED)
2. BCMRule      -- Bienenstock, Cooper & Munro, J Neurosci 1982;
                   threshold dynamics as in Shouval/Cooper tradition
3. USRule       -- Urbanczik & Senn, Neuron 2014 (dendritic prediction)
4. CalciumRule  -- calcium-control hypothesis: Lisman PNAS 1989;
                   Shouval, Bear & Cooper, PNAS 2002; Graupner & Brunel,
                   PNAS 2012; data-constrained for neocortical PCs by
                   Chindemi et al., Nat Commun 2022

Running this file produces sign_maps.png: instantaneous dw_bas over the
(x_ap, x_pre) plane for each rule, with slow variables frozen at their
baseline values.
"""

from dataclasses import dataclass
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------
# the neuron: apical gain modulation
# ----------------------------------------------------------------------

@dataclass
class GainNeuron:
    """x_som = gain(x_ap) * x_bas  (Larkum, Senn & Luscher 2004)."""
    g0: float = 1.0           # gain with apical silent
    g_max: float = 2.5        # gain with apical saturated
    theta_g: float = 1.0      # apical activation threshold of the gain
    sigma_g: float = 0.25     # sharpness
    x_ap0: float = 0.5        # resting apical activity

    def gain(self, x_ap):
        return self.g0 + (self.g_max - self.g0) * sigmoid(
            (x_ap - self.theta_g) / self.sigma_g)

    def x_som(self, x_bas, x_ap):
        return self.gain(x_ap) * x_bas


# ----------------------------------------------------------------------
# 1. burst-dependent rule (Payeur et al. 2021) -- recommended
# ----------------------------------------------------------------------

@dataclass
class BurstRule:
    """dw = eta * x_pre * x_som * (P - Pbar).

    P is the burst fraction, read off apical activity (BAC firing) with a
    somatic-rate term (critical frequency for Ca2+-spike ignition):
    Kampa/Letzkus/Stuart 2006; Sjostrom & Hausser 2006. The slow average
    Pbar makes the rule self-stabilizing (fixed point P = Pbar).
    """
    eta: float = 2e-2
    P_min: float = 0.02
    P_max: float = 0.60
    theta_P: float = 1.0      # apical threshold for Ca2+-spike/bursting
    sigma_P: float = 0.20
    kappa: float = 0.006      # 1/Hz, somatic-rate contribution
    tau_avg: float = 120.0    # s, running average of the burst fraction
    Pbar: float = None        # state; auto-initialized on first step

    def P(self, x_ap, x_som):
        return self.P_min + (self.P_max - self.P_min) * sigmoid(
            (x_ap + self.kappa * x_som - self.theta_P) / self.sigma_P)

    def step(self, x_pre, x_bas, x_som, x_ap, dt):
        P = self.P(x_ap, x_som)
        if self.Pbar is None:
            self.Pbar = P
        self.Pbar += dt / self.tau_avg * (P - self.Pbar)
        return dt * self.eta * np.asarray(x_pre) * x_som * (P - self.Pbar)


# ----------------------------------------------------------------------
# 2. BCM rule
# ----------------------------------------------------------------------

@dataclass
class BCMRule:
    """dw = eta * x_pre * x_som * (x_som - theta_M),
    tau_theta dtheta_M/dt = x_som^2 / E0 - theta_M.

    Purely somatic. NOTE: in this gain architecture BCM is not fully
    apical-blind -- x_ap scales x_som multiplicatively, so strong apical
    activity can push x_som across theta_M. But at fixed x_som, apical
    activity has no influence on the sign (contrast rule 1).
    The sliding threshold is metaplasticity and must be much slower than
    any induction protocol (Kirkwood, Rioult & Bear, Nature 1996).
    """
    eta: float = 3e-5
    E0: float = 2.3           # theta_M fixed point scale: theta = <x_som^2>/E0
    tau_theta: float = 2e4    # s (hours)
    theta_M: float = None     # state

    def step(self, x_pre, x_bas, x_som, x_ap, dt):
        if self.theta_M is None:
            self.theta_M = x_som ** 2 / self.E0
        self.theta_M += dt / self.tau_theta * (x_som ** 2 / self.E0 - self.theta_M)
        return dt * self.eta * np.asarray(x_pre) * x_som * (x_som - self.theta_M)


# ----------------------------------------------------------------------
# 3. Urbanczik-Senn rule
# ----------------------------------------------------------------------

@dataclass
class USRule:
    """Basal synapses learn to PREDICT the somatic rate; apical input is
    the teacher that nudges the soma away from the prediction.

    Prediction of the soma from basal drive alone (resting gain g_rest):
        x_som_hat = g_rest * x_bas
    dw = eta * x_pre * (x_som - x_som_hat)
       = eta * x_pre * x_bas * (gain(x_ap) - g_rest)

    i.e. in this architecture the US rule reduces to: plasticity tracks
    the DEVIATION OF THE APICAL GAIN FROM ITS RESTING VALUE. Plasticity
    stops when the prediction matches (self-limiting), and there is no
    somatic-rate-only plasticity (contrast rules 1, 2, 4).
    """
    eta: float = 2e-3
    g_rest: float = None      # resting gain; set from the neuron

    def bind(self, neuron: GainNeuron):
        self.g_rest = neuron.gain(neuron.x_ap0)
        return self

    def step(self, x_pre, x_bas, x_som, x_ap, dt):
        return dt * self.eta * np.asarray(x_pre) * (x_som - self.g_rest * x_bas)


# ----------------------------------------------------------------------
# 4. calcium-threshold rule
# ----------------------------------------------------------------------

@dataclass
class CalciumRule:
    """Calcium-control hypothesis, rate-based reduction.

    Per-synapse calcium proxy with two sources:
      NMDA:  needs pre & postsynaptic depolarization -> alpha*x_pre*x_som
      VDCC / dendritic Ca2+ spikes: recruited by apical activity during
             somatic firing -> beta*x_som*sigmoid(x_ap)

        c_i = alpha * x_pre_i * x_som + beta * x_som * s(x_ap)

    Two thresholds theta_d < theta_p (Lisman 1989):
      c < theta_d           -> no change
      theta_d < c < theta_p -> LTD
      c > theta_p           -> LTP

        dw_i = eta * [ gamma_p * H(c_i - theta_p) - gamma_d * H(c_i - theta_d) ]

    (H = smooth sigmoid). Depression and potentiation rates gamma_d,
    gamma_p as in Graupner & Brunel 2012; parameters data-constrained
    for neocortical pyramidal connections by Chindemi et al. 2022.
    """
    eta: float = 5e-3
    alpha: float = 0.02       # NMDA coincidence gain (1/Hz)
    beta: float = 0.15        # apical Ca2+-spike contribution (1/Hz)
    x_half: float = 2.0       # Hz; presynaptic gating of the Ca2+-spike
                              # term keeps the rule synapse-specific
                              # (ungated it is a heterosynaptic component,
                              #  which does exist: Golding et al. 2002)
    theta_ap: float = 1.0     # apical threshold for Ca2+-spike recruitment
    sigma_ap: float = 0.20
    theta_d: float = 1.0      # LTD threshold
    theta_p: float = 3.0      # LTP threshold
    gamma_d: float = 0.5      # depression rate
    gamma_p: float = 1.0      # potentiation rate
    smooth: float = 0.15      # softness of the threshold functions

    def calcium(self, x_pre, x_som, x_ap):
        x_pre = np.asarray(x_pre)
        gate = x_pre / (x_pre + self.x_half)
        return (self.alpha * x_pre * x_som
                + self.beta * gate * x_som
                  * sigmoid((x_ap - self.theta_ap) / self.sigma_ap))

    def step(self, x_pre, x_bas, x_som, x_ap, dt):
        c = self.calcium(x_pre, x_som, x_ap)
        pot = self.gamma_p * sigmoid((c - self.theta_p) / self.smooth)
        dep = self.gamma_d * sigmoid((c - self.theta_d) / self.smooth)
        return dt * self.eta * (pot - dep)


# ----------------------------------------------------------------------
# demo: instantaneous sign maps over the (x_ap, x_pre) plane
# ----------------------------------------------------------------------

def sign_maps(fname="sign_maps.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    TXT, GRID = "#33322e", "#e5e4df"
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": TXT, "text.color": TXT,
        "xtick.color": TXT, "ytick.color": TXT, "font.size": 9.5,
        "axes.titlesize": 10, "figure.dpi": 150, "savefig.bbox": "tight",
    })

    neuron = GainNeuron()
    x_ap = np.linspace(0.0, 2.0, 220)
    x_pre = np.linspace(0.0, 30.0, 220)
    XA, XP = np.meshgrid(x_ap, x_pre)
    w = 1.0
    XB = w * XP                            # x_bas (single synapse)
    XS = neuron.gain(XA) * XB              # x_som

    # baseline operating point freezes the slow variables
    xp0 = 5.0
    xs0 = neuron.gain(neuron.x_ap0) * w * xp0

    burst = BurstRule()
    burst.Pbar = burst.P(neuron.x_ap0, xs0)
    bcm = BCMRule()
    bcm.theta_M = xs0 ** 2 / bcm.E0
    us = USRule().bind(neuron)
    ca = CalciumRule()

    maps = [
        ("burst-dependent\n(Payeur et al. 2021)",
         burst.eta * XP * XS * (burst.P(XA, XS) - burst.Pbar)),
        ("BCM\n(Bienenstock et al. 1982)",
         bcm.eta * XP * XS * (XS - bcm.theta_M)),
        ("Urbanczik–Senn\n(2014)",
         us.eta * XP * (XS - us.g_rest * XB)),
        ("calcium thresholds\n(Shouval 2002; Graupner & Brunel 2012)",
         ca.eta * (ca.gamma_p * sigmoid((ca.calcium(XP, XS, XA) - ca.theta_p) / ca.smooth)
                   - ca.gamma_d * sigmoid((ca.calcium(XP, XS, XA) - ca.theta_d) / ca.smooth))),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4), sharey=True)
    for ax, (title, Z) in zip(axes, maps):
        # asymmetric diverging norm: white stays at 0, each sign uses its
        # own range so weaker LTD lobes remain visible
        vmin = min(np.nanmin(Z), -1e-12)
        vmax = max(np.nanmax(Z), 1e-12)
        pc = ax.pcolormesh(XA, XP, Z, cmap="RdBu_r",
                           norm=TwoSlopeNorm(0, vmin=vmin, vmax=vmax),
                           rasterized=True)
        ax.contour(XA, XP, Z, levels=[0], colors=TXT, linewidths=0.9,
                   linestyles="--")
        ax.axvline(neuron.x_ap0, color=TXT, lw=0.8, ls=":")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("apical activity $x_{ap}$")
        cb = fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("$\\mathrm{d}w_{bas}/\\mathrm{d}t$", fontsize=8)
    axes[0].set_ylabel("presynaptic rate $x_{pre}$ (Hz)")
    fig.suptitle("instantaneous basal plasticity over the (apical, presynaptic) plane"
                 "  —  red: LTP, blue: LTD, dashed: $\\dot w = 0$, dotted: resting $x_{ap}$",
                 fontsize=9.5, y=1.04)
    fig.tight_layout()
    fig.savefig(fname)
    print(f"wrote {fname}")


if __name__ == "__main__":
    sign_maps()
