# Bottom-up plasticity sandbox

Throwaway code for trying out basal (bottom-up) plasticity rules and tuning their hyperparameters without touching
the `l5apical` package. Delete this directory when done; nothing outside it depends on it. Results are written to
`sandbox/results/` which is gitignored.

## Layout

| file | purpose |
|---|---|
| `rules.py` | the candidate rules (all with the same signature, registered in `RULES`), their default hyperparameters (`RULE_HP`) and the weight-bounding mechanisms (`apply_bound`) |
| `plasticity_rules_tmp.py` | the literature rules as originally written in generic rate-model variables; `rules.py` maps them onto this model (see the module docstring there) |
| `simulate.py` | mirror of `l5apical.simulations.simulate_seed` with a rule hook, hyperparameter dict, optional apical inhibition, w_bas snapshots, and cheap Python metrics (no Matlab needed) |
| `sweep.py` | CLI: run a single config or a cartesian grid, parallel over (config, seed), writes pickles + `summary.csv` |
| `plot.py` | CLI: one overview figure per config plus a cross-config comparison |
| `sign_maps.py` | CLI: instantaneous update of every rule over the (apical gain, basal weight) plane |
| `show.py`, `tab.py` | print sweep summaries (`show` the full table, `tab` a compact ranked one with one column per varied hyperparameter) |
| `compare.py` | overlay performance and detector counts of several configurations in one figure |
| `diag.py` | one-seed diagnostic of the apical loop (TD error, `x_pre_t`, apical weights, gains, texture weights) |
| `diag_gated.py` | one-seed diagnostic of `us` / `burst`: post-synaptic factor at texture time by texture, update mass per stimulus class, weight trajectories of the most texture-selective neurons |
| `diag_gh.py` | one-seed diagnostic of `gated_hebb`: post-synaptic factor `gate * (post - theta)` at tone and texture time (fraction of neurons potentiated, T1 vs T2), potentiation and depression mass per stimulus class, weight trajectories and detector counts |
| `diag_compare.py` | one-seed diagnostic with the same measures for every rule: where potentiation and depression go (row class x step type), across-neuron concentration of potentiation, sign of the post-synaptic factor vs the neuron's own drive, share of potentiation on the strongest synapse, detectors over time |
| `diag_tone_ablation.py` | causal ablation: `us` / `burst` at diverging basal rates with the tone row's basal update zeroed, 5 seeds each |
| `diag_gated_ablation.py` | `us` / `burst` with the running average optionally ignoring the outcome step (causal test only); per-block gain, baseline and sign statistics |
| `check_reproduces.py` | asserts the mirror is bit-identical to the main code for `DEFAULT`, `APICAL_INHIBITION` and `BU_PLASTICITY` |

Run everything in the repository's own conda env `tdd`, built as the top-level README describes
(`conda create -n tdd python=3.10.8`, then `pip install -r requirements.txt`; python at `~/miniconda3/envs/tdd/bin/python`).
Do not use other envs on the machine.

## Usage

```
python -m sandbox.check_reproduces            # run after any change to simulate.py or the main trial loop
python -m sandbox.sign_maps                   # sandbox/results/sign_maps.png
python -m sandbox.sweep --name rules --set init_power=12 --grid '{"rule": ["burst", "us", "bcm"], "bound": ["soft", "l1"], "lr_bas": [0.016, 0.064]}' --seeds 5 --workers 8
python -m sandbox.sweep --name rules_inh --set init_power=12 n_trials=4000 n_trials_inhibited=1800 --grid '{"rule": ["burst", "us"]}' --seeds 5
python -m sandbox.sweep --name baseline_default --set rule=none init=identity n_noise=200 --seeds 5   # the paper's default model
python -m sandbox.show --rank hp_burst                   # summary table, best configurations first
python -m sandbox.tab m3_final10 m3_inh                 # compact ranked table, one column per varied hyperparameter
python -m sandbox.compare out.png p12_rules/<config> p12_inh/<config>   # overlay performance and detector counts
python -m sandbox.plot rules
```

`--set key=value` overrides a default from `DEFAULT_HP` in `simulate.py` (which includes `RULE_HP`); `--grid` takes
a JSON dict or a path to one. A 1800-trial seed takes about 20 s on one core; a 4000-trial apical-inhibition seed
about 45 s. Results are slimmed by default (`store_full=False` drops the per-trial arrays no sandbox metric uses).
A run whose value estimate diverges (NaN) is stopped and returned with what was recorded so far; the summary
column `diverged` counts such seeds.

## Hyperparameters (`DEFAULT_HP`)

- `rule`, `bound`, `lr_bas`: the rule, how its update is bounded (`soft` = the `(1 - w) w` factor of the main code,
  `l1` = per-neuron renormalization to sum 1, `clamp`, `none`), and the basal learning rate (the main code uses the
  common `LR` = 0.016).
- `lr_ap`, `lr_policy`, `lr_td`, `lr_trace`: learning rates of the apical weights, the agent's policy network, the
  TD value estimator and the apical afferent trace `x_pre_t` (main code: all four equal `LR`).
- `theta_bas_0`: threshold scale seen by the basal rule only (`None`, the default, means the basal rule uses the
  apical thresholds; a value builds separate basal thresholds `theta_bas_0 * expected drive + BKG`, leaving the apical
  rule's `theta_0` untouched). At `init_power` 12 a value of 16 puts the threshold at about 1.5 times the median
  neuron's strongest un-gained response.
- `rule_sees_gain`: `False` hands the basal rule the un-gained somatic rate and resting apical variables while the
  network keeps the real gain (causal ablation of the implicit apical route through the soma; default `True`).
- `lr_dep`, `dep_form`: an extra subtractive depression term added to any rule's update (`lr_dep` 0, the default, is
  off). `thr` is the depression half of `hebb`, `- lr_dep * x_in * relu(theta_bas - x_som)` with the basal threshold
  from `theta_bas_0`, so every synapse active on a sub-threshold neuron is depressed by the gap and neurons above
  threshold are spared; `const` depresses every active synapse by `lr_dep`. With `rule = none` it is a pruning-only
  baseline. Used to test whether the rules with an explicit apical factor (`us`, `burst`) work once they can prune like `hebb`.
- `theta_gate`: per-neuron gate applied to any rule's update after the rule is evaluated (`None`, the default, is
  off): `pot` lets a neuron potentiate only while `x_som` exceeds its basal threshold (depression is untouched),
  `all` switches the basal plasticity of sub-threshold neurons off entirely. Used to test whether a rule with an
  explicit apical factor (`us`, `burst`) works once its potentiation is restricted to strongly driven neurons, as in `hebb` with
  `theta_bas_0`.
- `n_noise`, `n_z`, `init`, `init_power`, `theta_0`: distractor count, neuron count, `random` (L1-normalized
  `rand ** init_power`, as in the mixed model) or `identity` initialization, and the threshold scale. Larger
  `init_power` gives sparser initial weights; the median strongest weight of a neuron is about 0.19 at power 6, 0.28
  at 12 and 0.4 at 24 (with 33 stimuli).
- `n_trials`, `n_trials_inhibited`: apical dendrites are silenced for the first `n_trials_inhibited` trials exactly
  as in `Simulation.APICAL_INHIBITION` (main code: 1800 of 4000).
- rule-specific keys, documented in `rules.py`: `gate_power`, `gh_post`, `gh_theta_scale`, `burst_tau`, `burst_kappa`, `bcm_tau`, `bcm_e0_scale`, `us_tau`, `ca_beta`,
  `ca_theta_p`, `ca_ltd_frac`, `ca_gamma_d`, `ca_smooth`, `gate_power`, `oja_decay`. JSON `Infinity` is accepted for
  the time constants (= fixed threshold / resting-gain prediction).

## Rules

Variables: `x_in` binary stimuli, `x_bas` basal drive, `x_ap` apical activation (`apical_transfer`, 0.1 at rest),
`gain` in [1, 10], `x_som = gain * x_bas`, `theta` the per-neuron threshold of the mixed model.

- `none`: control.
- `hebb`: `x_in (x_som - theta)`, the rule in the main code. No explicit apical factor; the apical signal enters through the gain in `x_som` (theta is far below any driven response).
- `gated_hebb`: `hebb` times the normalized apical activation (three-factor).
- `burst` (Payeur et al. 2021): `x_in x_som (P - Pbar)` with the burst fraction `P = x_ap` and a running average
  `Pbar`. Off when the apical dendrite is silent.
- `bcm` (Bienenstock et al. 1982): `x_in x_som (x_som - theta_M)` with a sliding threshold whose fixed point at
  initialization equals theta. No explicit apical factor; apical influence only through the gain in `x_som`.
- `us` (Urbanczik & Senn 2014): `x_in x_bas (gain - g_bar)` with `g_bar` a running average of the gain (`us_tau`;
  inf = resting gain 1, then purely potentiating). Off when the apical dendrite is silent.
- `calcium` (Shouval et al. 2002, Graupner & Brunel 2012): calcium `x_in x_som (1 + ca_beta * apical)` with an LTD
  window below and LTP above a threshold set relative to each neuron's strongest initial input.
- `oja`: Oja's rule, needs `bound = clamp` or `none` and a small `lr_bas`.

## Metrics (per seed, in `summary.csv` and the figures)

- `expert_proxy`: trial after which the 100-trial moving performance stays above 0.8 (stands in for the Matlab
  Smith et al. expert trial; NaN if never reached). For the default model this gives about 1545 trials (5 seeds).
- `expert_after_lift`, `perf_inhibited_end`: with apical inhibition, the expert proxy counted from the lift and the
  mean performance over the last 200 inhibited trials (default model: 1549 and 0.50).
- `task_frac_*`: fraction of a neuron's basal weight coming from the three task stimuli, averaged over neurons.
- `selectivity_*`: |w_T1 - w_T2| / (w_T1 + w_T2), averaged over neurons. Stimulus rows `T1_IDX` = 2 and `T2_IDX` = 1.
- `w_bas_change`, `w_bas_sum_mean`: mean absolute change of the basal weights, and mean total basal weight per neuron.

The stimulus ordering in `w_bas` rows is: 0 tone cue, 1 texture T2, 2 texture T1, then distractors.

## Findings so far

See the notes at the end of `CLAUDE.md`.
