# Bottom-up plasticity sandbox

Throwaway code for trying out basal (bottom-up) plasticity rules and tuning their hyperparameters without touching
the `l5apical` package. Delete this directory when done; nothing outside it depends on it. Results are written to
`sandbox/results/` which is gitignored.

## Layout

| file | purpose |
|---|---|
| `rules.py` | the candidate rules, all with the same signature, registered in `RULES` |
| `simulate.py` | verbatim mirror of `l5apical.simulations.simulate_seed` with a rule hook, hyperparameter dict, w_bas snapshots, and cheap Python metrics (no Matlab needed) |
| `sweep.py` | CLI: run a single config or a cartesian grid, parallel over (config, seed), writes pickles + `summary.csv` |
| `plot.py` | CLI: one overview figure per config plus a cross-config comparison |
| `check_reproduces.py` | asserts the mirror is bit-identical to the main code for `Simulation.DEFAULT` and `Simulation.BU_PLASTICITY` |

## Usage

```
python -m sandbox.check_reproduces            # run after any change to simulate.py or the main trial loop
python -m sandbox.sweep --name rules --grid '{"rule": ["soft_bounded", "l1_renormalized", "bcm", "apical_gated"]}' --seeds 5 --workers 6
python -m sandbox.sweep --name lr --set rule=l1_renormalized --grid '{"lr_bas": [0.002, 0.004, 0.008, 0.016]}' --seeds 5
python -m sandbox.plot rules
```

`--set key=value` overrides a default from `DEFAULT_HP` in `simulate.py`; `--grid` takes a JSON dict or a path
to one. Rule-specific hyperparameters (`oja_decay`, `gate_power`) are passed the same way. A full 1800-trial seed
takes about 90 s on one core, so a 4 config x 5 seed sweep on 6 workers is roughly 5 minutes.

## Rules

- `soft_bounded`: the rule in the main code, Hebbian x (post - theta) x (1 - w) w.
- `hard_clamped`: same without the soft bound. **Diverges** at the default learning rate: total drive per neuron
  grows without bound, the TD estimate explodes and the policy produces NaN.
- `l1_renormalized`: Hebbian step, then each neuron's afferents renormalized to sum to 1 (competitive).
- `oja`: Oja's rule. **Also diverges** at the default learning rate for the same reason; needs a much smaller `lr_bas`.
- `bcm`: post (post - theta) post-synaptic factor, soft bounded.
- `apical_gated`: three-factor rule gated by the normalized apical gain.
- `none`: control, no basal plasticity.

Add a rule by writing a function in `rules.py` with the shared signature and adding it to `RULES`.

## Metrics (per seed, in `summary.csv` and the figures)

- `expert_proxy`: trial after which the 100-trial moving performance stays above 0.8. Stands in for the Matlab
  Smith et al. expert trial; NaN if never reached.
- `task_frac_*`: fraction of a neuron's basal weight coming from the three task stimuli, averaged over neurons.
- `selectivity_*`: |w_T1 - w_T2| / (w_T1 + w_T2), averaged over neurons. Stimulus rows `T1_IDX` = 2 and `T2_IDX` = 1.
- `w_bas_change`: mean absolute change of the basal weights from init to end.

The stimulus ordering in `w_bas` rows is: 0 tone cue, 1 texture T2, 2 texture T1, then distractors.
