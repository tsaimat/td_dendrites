"""
Assert that the ported main code (l5apical.simulations.simulate_seed) reproduces the sandbox mirror bit for bit at the
settings of the figure sets it was ported from: control4 / hebb4 / bcm4 / calcium4 / apical4 (biological regime) and
control3 / hebb3 / bcm3 / calcium3 (fast-learning regime, where the apical rule has no setting), each without and with
apical inhibition, on short runs.
Run:  python -m sandbox.check_reproduces [n_trials]
"""
import json
import sys
import numpy as np
import torch
import l5apical.helper as H
import l5apical.simulations as S
from l5apical.helper import *
from random import seed as rd_seed
from sandbox.simulate import simulate_seed_bu, get_params_bu, DEFAULT_HP
from sandbox.figures import FIG_DIR

CASES = [(Simulation.DEFAULT, Simulation.APICAL_INHIBITION, 'control'), (Simulation.BU_HEBB, None, 'hebb'),
         (Simulation.BU_BCM, None, 'bcm'), (Simulation.BU_CALCIUM, None, 'calcium'),
         (Simulation.BU_APICAL, None, 'apical')]


def initial_w_bas(hp: dict, seed: int) -> np.ndarray:
    """The sandbox's initial basal weights (after the neuron relabeling), drawn as at the start of a run."""
    rd_seed(seed); np.random.seed(seed=seed); torch.manual_seed(seed=seed)
    return get_params_bu({**DEFAULT_HP, **hp, 'sort_neurons': True})[5].numpy()


def main(n_trials=120, seed=1):
    H.N_TRIALS, H.N_TRIALS_AP_INH, S.N_TRIALS = n_trials, n_trials, n_trials // 2  # inhibition lifted half-way
    ok = True
    for fast, suffix in ((False, '4'), (True, '3')):
        for sim, sim_inh, name in CASES:
            if sim in BU_SIMULATIONS and sim not in get_bu_simulations(fast=fast):
                continue
            hp = json.load(open(FIG_DIR / f'{name}{suffix}' / 'hp.json'))
            for inhibition in (False, True):
                n_inh = n_trials // 2 if inhibition else 0
                out = simulate_seed_bu({**hp, 'n_trials': n_trials, 'n_trials_inhibited': n_inh, 'store_full': True,
                                        'sort_neurons': True, 'gain_at_preferred': True}, seed=seed)
                out[K_W_BAS_FINAL], out[K_W_BAS] = out[K_W_BAS], initial_w_bas(hp, seed)
                if sim_inh is not None and inhibition:
                    ref = S.simulate_seed(sim_inh, seed=seed, fast=fast)
                else:
                    ref = S.simulate_seed(sim, seed=seed, fast=fast, inhibition=inhibition and sim_inh is None)
                shared = [k for k in ref if k in out]
                bad = [k for k in shared if not np.array_equal(np.asarray(ref[k], float), np.asarray(out[k], float),
                                                               equal_nan=True)]
                label = f"{name}{suffix} {'inhibited' if inhibition else 'fresh'}"
                only_main = [k for k in ref if k not in out]  # Recorded by the main code only (not by the mirror)
                print(f"{label:22s}: {'IDENTICAL' if not bad else 'DIFFERS in ' + ', '.join(bad)} ({len(shared)} keys"
                      + (f"; not in the mirror: {', '.join(only_main)}" if only_main else '') + ')')
                ok &= not bad
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
