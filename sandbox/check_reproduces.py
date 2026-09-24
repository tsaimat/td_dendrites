"""
Assert that the sandbox loop is a faithful mirror of l5apical.simulations.simulate_seed.
Run:  python -m sandbox.check_reproduces [n_trials]
"""
import sys
import numpy as np
import l5apical.simulations as S
from l5apical.helper import *
from sandbox.simulate import simulate_seed_bu


def _short(n):
    gp = S.get_params
    def g(simulation, theta_0):
        r = list(gp(simulation, theta_0)); r[8] = n; return tuple(r)
    return gp, g


def main(n_trials=100):
    gp, g = _short(n_trials)
    cases = [
        (Simulation.BU_PLASTICITY, dict(rule='soft_bounded')),
        (Simulation.DEFAULT, dict(rule='none', init='identity', n_noise=N_Z - 3)),
    ]
    ok = True
    for sim, hp in cases:
        S.get_params = g
        ref = S.simulate_seed(sim, seed=0)
        S.get_params = gp
        out = simulate_seed_bu({**hp, 'n_trials': n_trials}, seed=0)
        bad = [k for k in ref if not np.array_equal(np.asarray(ref[k], float), np.asarray(out[k], float), equal_nan=True)]
        print(f"{sim.name:14s} vs sandbox {hp}: {'IDENTICAL' if not bad else 'DIFFERS in ' + ', '.join(bad)}")
        ok &= not bad
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
