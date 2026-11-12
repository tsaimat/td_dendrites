"""Checks of the gain-blind policy and apical replay knobs of sandbox.simulate (short runs, seed 0):
1. policy_sees_gain False without bottom-up plasticity behaves exactly as apical inhibition (the main code's
   APICAL_INHIBITION run, whose gain is 1 at every step while inhibited): identical trial outcomes.
2. Replaying a run's own recorded apical activity (record_apical, then apical_replay) reproduces that run exactly with
   the apical rule (apical4), which reads the gain and the un-gained basal drive only.
Run:  python -m sandbox.check_gainless [n_trials]"""
import sys, tempfile
from pickle import load
import numpy as np, torch
from l5apical.helper import *
from sandbox.simulate import simulate_seed_bu, K_W_BAS_H
from sandbox.scan_gainless import BASE

APICAL4 = dict(rule='apdev', apdev_tau=float('inf'), apdev_mult='bas', apdev_xbar='step', apdev_bas_power=2,
               bound='l1', lr_bas=0.005)


def main(n_trials=300, seed=0):
    torch.set_num_threads(1)
    ok = True
    blind = simulate_seed_bu({**BASE, 'n_trials': n_trials, 'policy_sees_gain': False}, seed=seed)
    with open(get_path(Simulation.APICAL_INHIBITION), 'rb') as handle:
        inhibited = load(handle)[seed][K_OUTCOME][:n_trials]
    same = np.array_equal(blind[K_OUTCOME], inhibited)
    print(f'gain-blind policy = apical inhibition: {"IDENTICAL" if same else "DIFFERS"} outcomes ({n_trials} trials)')
    ok &= same
    with tempfile.TemporaryDirectory() as tmp:
        recorded = simulate_seed_bu({**BASE, **APICAL4, 'n_trials': n_trials, 'record_apical': tmp}, seed=seed)
        replayed = simulate_seed_bu({**BASE, **APICAL4, 'n_trials': n_trials, 'apical_replay': tmp}, seed=seed)
    same = (np.array_equal(recorded[K_OUTCOME], replayed[K_OUTCOME])
            and np.array_equal(recorded[K_W_BAS_H], replayed[K_W_BAS_H]))
    print(f'replay of its own apical activity (apical4): {"IDENTICAL" if same else "DIFFERS"} outcomes and w_bas')
    ok &= same
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
