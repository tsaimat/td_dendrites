"""
Candidate bottom-up plasticity rules for the basal weights w_bas (shape: n_stimuli x n_neurons).

Every rule has the same signature and returns the updated weight matrix:

    rule(w_bas, x_in, x_som, x_bas, gains, thetas, hp) -> w_bas

    w_bas   current basal weights (n_stim, n_z)
    x_in    binary vector of currently active stimuli (n_stim,)
    x_som   somatic activity of the pyramidal neurons (n_z,)
    x_bas   basal activation before apical gain, including background (n_z,)
    gains   multiplicative apical gains (n_z,)
    thetas  per-neuron plasticity thresholds (n_z,)
    hp      hyperparameter dict; every rule reads hp['lr_bas'], rule-specific keys are documented per rule

Add a new rule by writing a function and registering it in RULES.
"""
import torch
from l5apical.helper import MAX_GAIN, BKG


def _post(x_som, thetas):
    """Thresholded post-synaptic factor shared by several rules."""
    return x_som - thetas


def soft_bounded(w_bas, x_in, x_som, x_bas, gains, thetas, hp):
    """
    The rule currently in l5apical.simulations: Hebbian with a thresholded post-synaptic term and the soft bound
    (1 - w) * w that keeps weights in [0, 1] and freezes weights at exactly 0 or 1.
    """
    dw = torch.outer(hp['lr_bas'] * x_in, _post(x_som, thetas)) * (1 - w_bas) * w_bas
    return torch.clamp(w_bas + dw, min=0, max=1)


def hard_clamped(w_bas, x_in, x_som, x_bas, gains, thetas, hp):
    """Same Hebbian term without the soft bound; weights are simply clamped to [0, 1]."""
    dw = torch.outer(hp['lr_bas'] * x_in, _post(x_som, thetas))
    return torch.clamp(w_bas + dw, min=0, max=1)


def l1_renormalized(w_bas, x_in, x_som, x_bas, gains, thetas, hp):
    """
    Hebbian threshold step followed by per-neuron L1 renormalization, so every neuron keeps a fixed total amount
    of basal weight (competition between afferents). Matches the init, whose columns sum to 1.
    """
    dw = torch.outer(hp['lr_bas'] * x_in, _post(x_som, thetas))
    w = torch.clamp(w_bas + dw, min=0)
    return torch.nn.functional.normalize(w, p=1., dim=0)


def oja(w_bas, x_in, x_som, x_bas, gains, thetas, hp):
    """
    Oja's rule: Hebbian growth minus a decay proportional to w * post^2, which bounds the L2 norm of each neuron's
    afferents without an explicit clamp. Uses the raw somatic activity as post-synaptic factor.
    hp['oja_decay'] scales the decay term (1.0 is the textbook rule).
    """
    post = x_som
    dw = hp['lr_bas'] * (torch.outer(x_in, post) - hp.get('oja_decay', 1.0) * w_bas * post.pow(2))
    return torch.clamp(w_bas + dw, min=0, max=1)


def bcm(w_bas, x_in, x_som, x_bas, gains, thetas, hp):
    """
    BCM-shaped rule: post-synaptic factor post * (post - theta), so weak activity depresses and strong activity
    potentiates, with a sliding threshold replaced by the fixed per-neuron thetas. Soft bounded like the default.
    """
    post = x_som * _post(x_som, thetas)
    dw = torch.outer(hp['lr_bas'] * x_in, post) * (1 - w_bas) * w_bas
    return torch.clamp(w_bas + dw, min=0, max=1)


def apical_gated(w_bas, x_in, x_som, x_bas, gains, thetas, hp):
    """
    Three-factor rule: the Hebbian threshold term is gated by the apical gain, normalized to [0, 1], so basal
    synapses only change on neurons whose apical dendrite is currently receiving the top-down salience signal.
    hp['gate_power'] sharpens (>1) or softens (<1) the gate.
    """
    gate = ((gains - 1) / (MAX_GAIN - 1)).clamp(0, 1).pow(hp.get('gate_power', 1.0))
    dw = torch.outer(hp['lr_bas'] * x_in, gate * _post(x_som, thetas)) * (1 - w_bas) * w_bas
    return torch.clamp(w_bas + dw, min=0, max=1)


def none(w_bas, x_in, x_som, x_bas, gains, thetas, hp):
    """No bottom-up plasticity (control)."""
    return w_bas


RULES = {
    'none': none,
    'soft_bounded': soft_bounded,
    'hard_clamped': hard_clamped,
    'l1_renormalized': l1_renormalized,
    'oja': oja,
    'bcm': bcm,
    'apical_gated': apical_gated,
}
