# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Simulation code and figure generation for the manuscript "Error representations in apical dendrites of neocortical layer
5 neurons during learning" (Schoenfeld et al., bioRxiv 10.1101/2021.12.28.474360). A mouse agent learns a go/no-go
texture discrimination task; 1000 L5 pyramidal neurons receive mixed-selective basal input (lognormal weights from the
three task stimuli and 30 distractors) and a top-down TD-error signal onto their apical dendrites, which
multiplicatively gates somatic output. Variants add bottom-up (basal) plasticity (`hebb`, `bcm`, `calcium`, and
`apical`, gated by the apical gain, biological regime only) and a fast-learning regime. There is no test suite;
correctness is checked by regenerating the panels in `panels/` and comparing them to the manuscript figures, and the
trial loop against the sandbox mirror (`python -m sandbox.check_reproduces`).

## Environment and commands

Python 3.10, dependencies pinned in `requirements.txt` (torch 1.13, numpy 1.23, scipy 1.9, matplotlib 3.7,
statsmodels). Build the env as the README says (`conda create -n tdd python=3.10.8`, then `pip install -r
requirements.txt`) and use only that `tdd` env. No Matlab: `helper.smith_analysis()` ports the Smith et al. (2004)
estimator of the performance traces.

```
python main.py plot          # (default) plot all panels to panels/ (fast regime: panels/fast/), with svg/ copies
python main.py simulate      # rerun every simulation, including the performance traces (about 90 min on 8 cores)
python main.py perf          # extract the performance traces of the saved results again
```

`simulate` runs the 18 runs of `get_runs()`, then the `theta_0` sweeps of the biological and of the fast regime (the
fast one 7.4 min) and the learning-rate sweeps of the fast regime (54 runs, 42 min). Each `simulate_seeds()` call runs
its 10 seeds in
parallel processes with one torch thread each (results do not depend on the thread count), adds each seed's Smith
performance trace and learning and expert trials (about 20 s for a 1800-trial seed, 50 s for a 4000-trial one, more
than the simulation itself) and writes its pickle once, complete, so an interruption leaves the files written before
it intact.

To plot a single panel interactively, call its function directly, e.g. `python -c "from l5apical.panels import *;
f4h_performance(saving=False)"`. Every panel function takes `saving: bool`; `False` shows the figure instead of
writing it. The panels of figs. 4 and S11 to S13 also take `fast: bool`, which plots the fast-learning runs into
`panels/fast/` under the same file names (`plot_all_fast()` draws all of them, and `plot_all()` calls it). `python -m l5apical.panels` runs the ad-hoc block at the bottom of `panels.py` (currently plots everything
with `saving=False`).

To run one simulation: `l5apical.simulations.simulate_seeds(Simulation.X, fast=..., inhibition=...)` (`inhibition`
only for the `BU_*` members, `Simulation.APICAL_INHIBITION` is the inhibited control; `theta_0=...` gives a sweep point
of `DEFAULT`, and so does `lr=('lr_td', value)` for a learning rate).

## Data files

- `results/` holds one pickle per run, each a list of `N_SEEDS` (10) dicts keyed by the `K_*` constants of `helper.py`
  (`KEYS_1D` per-trial vectors, `KEYS_2D` trials x (time steps or neurons)). Biological regime: `results_default.pickle`
  (`DEFAULT`), `results_perturbed.pickle` (`APICAL_INHIBITION`), `results_{hebb,bcm,calcium,apical}.pickle` and
  `results_{hebb,bcm,calcium,apical}_perturbed.pickle` (`BU_*` without and with apical inhibition). `results/fast/`
  holds the same files except the two `apical` ones (`BU_APICAL` has no fast-regime setting) for the fast-learning
  regime (its `DEFAULT` and `APICAL_INHIBITION` are read by `panels/fast/` and S15), `results/theta_0/t_<theta_0>.pickle`
  the `theta_0` sweep of `DEFAULT` and `results/fast/theta_0/t_<theta_0>.pickle` that of the fast regime (2026-10-23;
  1 to 5 in steps of 0.5), and `results/fast/<lr>/<lr>_<value>.pickle` the learning-rate sweeps of the fast regime
  (2026-10-24: `lr_td`, `lr_policy`, `lr_ap` and `lr_trace`, each varied alone from 1/8 to 8 times its value in
  half-octave steps plus the quarter-octave steps next to it, `lr_trace` only up to 1; `get_learning_rates()`; the
  value of the regime itself is the default simulation, which S15e reads from `results/fast/theta_0/t_3.0.pickle`,
  identical to `results/fast/results_default.pickle` in every trial outcome; 1.8 MB per value, 97 MB in all).
  `get_path()` is the single source of truth for these paths.
- A run whose weights become non-finite (the TD value estimate diverging, so far only in the fast sweeps: in the
  `theta_0` sweep every seed at 1 to 2, 2 of 10 at 2.5; in the learning-rate sweeps see Figure review) stops at the
  start of the next trial: its later trials have no outcome (nan),
  `get_diverged()` of `panels.py` detects it, and the Smith estimate counts those trials as errors (so it is never
  expert). Before 2026-10-23 such a run crashed the sweep.
- Slim format (user, 2026-10-13): the trials x neurons arrays (`w_ap`, `gain`, `x_som_texture`, `x_som_cue`) are
  float32, the eligibility trace `z` is not stored, and the sweep files keep only `outcome`, `correct_trials`, `w_bas`,
  the last row of `w_ap` and the performance. `performance` is the Smith trace (one entry more than the trials),
  `learning_t` and `expert_t` are ints counted from 0 (nan if not reached). `w_bas` holds the initial basal weights (the
  S12 panels bin neurons by them), `w_bas_final` the final ones, and the `BU_*` runs add `w_bas_history` (snapshots
  after trials 0, 50, ..., last) and `w_bas_history_t`. `gain` is recorded at the time steps when each neuron's
  preferred stimulus (largest initial weight) is present, in every protocol. `x_som_texture` and `x_som_cue` are the
  somatic activity at the texture and at the cue time step; `x_som_cue` exists since 2026-10-16 (for the cue-selective
  neurons of f4f), so far only in `results_default.pickle` (re-simulated, every other key bit-identical), its fast twin
  `results/fast/results_default.pickle` (re-simulated 2026-10-23, every other key bit-identical, 307 MB) and the two
  `apical` runs (new on 2026-10-22); the other runs get it at the next `python main.py simulate` (about 1.7 GB more).
  Sizes: 307 MB (`DEFAULT`), 518 MB (`APICAL_INHIBITION`), 284 / 625 MB (`BU_*` without / with inhibition; 356 / 786 MB
  for `apical`, which has `x_som_cue`), 8.3 GB for the 18 runs, 1.8 MB per sweep value.
- The results are not stored in git (user, 2026-10-24): `/results/` is in `.gitignore`, and the LFS pointers of the
  pre-port `results_default.pickle` and `results_perturbed.pickle` were removed (history keeps them). `.gitattributes`
  still routes `*.pickle` to LFS as a guard; never `git add` a result (GitHub rejects files over 100 MB, and the
  history would carry them).
- Everything Matlab was removed on 2026-10-14 (user): the `Smith/` folder with the original Smith et al. scripts,
  `getperfs.m` and the `.mat` files of the trial outcomes and performances. Git history keeps them up to 390281e.
- The `clean` branch (the same code without results) is not on the fork: origin only has `main`.

## Architecture

`main.py` is a thin dispatcher into the `l5apical` package:

- `helper.py`: model constants (neuron and distractor counts, the lognormal init `INIT_SIGMA`, `INIT_SCALE`,
  `INIT_CLIP`, trial timing in time steps at `HZ` = 4 steps/s), `HYPERPARAMETERS` (`theta_0` and the learning rates of
  the policy, TD estimator, apical weights and apical afferent trace for the `biological` and `fast` regimes),
  `RULE_HYPERPARAMETERS` (bound, `lr_bas` and rule constants per regime and `BU_*` member; `BU_APICAL` has no
  fast-regime setting), the `Simulation` and `Outcome` enums, the apical transfer and gain functions, `get_path()` (maps
  a simulation and its `fast`, `inhibition`, `theta_0` and `lr` options to its results file), `get_bu_simulations()`
  (the `BU_*` members with a setting in a regime), `get_runs()` (the 18 runs), `get_run_name()`, `get_thetas_0()` (the
  sweep values of a regime), `LEARNING_RATES` and `get_learning_rates()` (the values of the learning-rate sweeps of the
  fast regime), and `smith_analysis()`, the
  Python port of the learning-curve estimator of Smith et al. (2004) (`runanalysisv3.m` and its helpers; exact
  logit-normal quantiles instead of the Monte Carlo ones; validated against the Matlab output: identical learning and
  expert trials, traces within 3e-3).
- `simulations.py`: `Agent` is the policy network that picks lick/no-lick from somatic activity and does a
  REINFORCE-style update on reward. `get_params()` draws the apical weights, the distractor probabilities and the
  lognormal basal weights (L1-normalized per neuron to `INIT_SCALE`, clipped at 1), sets the apical plasticity
  thresholds (`theta_0` times the expected basal drive per time step, plus `BKG`) and relabels the neurons so that 0, 1,
  2 are the best tone, T2 and T1 neurons (the panels show them as cue, no-go and go neuron). `simulate_seed()` is the
  per-seed trial loop: basal activation, apical gain from `w_ap * x_pre_t`, somatic output, TD value estimate and apical
  plasticity gated by the per-neuron thresholds; apical inhibition silences the outcome-time apical activity for the
  first `N_TRIALS` of `N_TRIALS_AP_INH` trials. The `BU_*` members apply `basal_rule()` (it reads the input, the basal
  drive before the gain, the somatic activity and the apical activation) and `bound_basal_weights()` at every stimulus
  time step and once at the outcome (without input: the thresholds of hebb, bcm and calcium see the outcome activity,
  `BU_APICAL` depresses each synapse by its input's running average times gain - 1 at the background drive, and the
  `pre` / `l1` bounds renormalize). The loop stops a run whose weights become non-finite (see Data files).
  `simulate_seeds()` runs the seeds in parallel processes
  (`simulate_and_analyse_seed()`), adds their performance (`get_performance()`) and pickles the list;
  `extract_performances()` redoes the performance of all saved results; `run()` does everything.
- `panels.py`: `get_results()` loads a pickle and optionally aligns all traces to each seed's expert trial
  (`align_to_expert()`: the `N_BEFORE_EXPERT` (500) trials before it, or `n_before`, and the `N_EXPERTS` (150) from
  it on, nan outside the simulation). `get_texture_responsive()` is the responsiveness test of S12d (mean somatic activity in the texture
  window above 0.1 over a range of trials), `get_texture_selective()` gives the go- and no-go-selective neurons (the
  selective bins of S12d, go minus no-go response beyond 0.2), `get_cue_selective()` the cue-selective neurons (cue
  response above 0.1 and above the texture response by more than 0.2), `get_sensory_dendrites()` and `get_gains()`
  compute the sensory-dendrite activity and the gain at a time step of any group of neurons from the stored `w_ap` and
  `x_pre_t` (exact to 1e-7 against the recorded go/no-go trace; gains exact at the cue and within 0.0002 on average at
  the texture), and `get_trials()` returns each seed's trials, all or aligned to its expert trial. Shared helpers
  (`plot_trace`, `plot_v_hat`, `plot_apical_raster`, `plot_apical_trace_boxplot`, `set_style`, `save_or_show`) are
  wrapped by thin per-panel functions named after manuscript figure panels (`f4c_...`, `fs12b_...`). `plot_all()` is the
  ordered list of every panel. All panels read the biological `DEFAULT` and `APICAL_INHIBITION` runs (S12 included, so
  `fs12a` plots the same as `f4c`), `fs12b_theta0` the sweep, and `fs14b` to `fs14f` (the bottom-up plasticity figure,
  draft, the apical rule only since 2026-10-23; panel a is the schematic) the biological `BU_APICAL` runs in both
  protocols (c against the control). The panels of fig. 4h, S12c and S12c/d are thin wrappers of helpers that take a
  `simulation` (`plot_inhibition_performance`, `plot_inhibition_expert_trials`, `plot_inhibition_w_ap`,
  `plot_selectivity_traces`, `plot_selectivity_distributions` with `phases`, `width`, `y_max` and the legend place);
  `fs14b` (the performance only), `fs14e` and `fs14f` call the first and the last two with `BU_APICAL`.
  `get_bu_results()` loads the control or a rule in either protocol, `get_bu_conditions()` lists the control and the
  rules of a regime, `get_recruited()` (final `w_ap` above 0.5), `get_basal_change()` (summed absolute change of each
  neuron's basal weights from the initial ones, per snapshot) and `get_input_class_weights()` (tone, preferred and other
  texture, distractor sum) serve the `plot_bu_*` helpers. Of those, `fs14c` and `fs14d` use
  `plot_bu_paired_expert_trials` and `plot_bu_recruited_weights`; the others (all rules against the control, with
  `fast=True` for the fast regime) have no wrappers since 2026-10-23, and the PDFs of the five-condition draft are in
  `panels/s14_rules_draft/` (gitignored). `get_texture_selectivity()` and `plot_selectivity_histogram()` are shared with
  `fs12cd`. The panels of figs. 4 and S11 to S13 (not S11b/c, which does not depend on the regime) take `fast` and then
  save into `panels/fast/` (`get_plot_dir()`, `FAST_DIR`); `plot_all_fast()` lists them. In the fast regime two
  panels adapt by rule, leaving the biological ones pixel-identical: `plot_apical_trace_boxplot` (f4g, S13) shows the
  whole 50-trial batches before the expert trial that every seed has (`n_before` 500 biological, 150 fast, whose
  earliest expert trial is 189) and skips the Tukey test when a learning phase has fewer than two seeds (only seed 4
  has one in the fast regime); `fs12b_theta0` (`plot_theta0_w_ap()`) leaves diverged seeds out and shades the
  `theta_0` values at which every seed diverged (`COL_DIVERGED`) or some did (`COL_PARTLY_DIVERGED`). It also marks
  the regime's `theta_0` with a dashed line in both regimes (user, 2026-10-24). S15 (draft, the
  fast regime; layout of the user, 2026-10-24, version 3): `fs15a` is fig. 4h's performance panel in the fast regime
  (`plot_inhibition_performance()` with `plot_dir`), `fs15b` fig. S14c's paired expert trials with the biological
  against the fast regime (`plot_paired_expert_trials()`, which `plot_bu_paired_expert_trials()` calls for S14c), `fs15c`
  fig. S12b's `theta_0` panel in the fast regime (`plot_theta0_w_ap()`, the same as the fast S12b replot), `fs15d`
  the expert trials of the fast `theta_0` sweep (`get_theta_0_sweep()`) and `fs15e`, two
  panels wide, those of the learning-rate sweeps (`get_learning_rate_sweep()`, `LR_LABELS`). d and e share
  `plot_expert_trials_sweep()` (a cross at the regime's value) and are each laid out with the other's decoration as a
  stand-in, replaced after `adjust_figure()` (d with a title, e with d's `theta_0` label), so that their axes have the
  same bottom and top and the y ticks align across the row. b reads both regimes (`get_regime_expert_trials()`,
  `REGIME_LABELS`, `REGIME_COLORS`: grey and `COL_FAST` `#2a78d6`, validated for color vision deficiencies against
  grey and `COL_RESTORED`). Nothing plots the hebb, bcm and calcium runs or the fast `BU_*` runs.
  Figure dimensions are in cm-derived constants at the top of the file and output is PDF at `DPI` plus an SVG copy.
- `parula.py`: Matlab parula colormap data.

Adding a new simulation variant means: an enum member in `Simulation` (and in `BU_SIMULATIONS` for a basal rule, which
`get_runs()` then runs in every regime where `RULE_HYPERPARAMETERS` has a setting for it), its name in `get_rule_name()`
/ `get_run_name()`, its constants in `RULE_HYPERPARAMETERS`, a branch in `basal_rule()` (and its state in
`simulate_seed()`), and for a basal rule its label and color in `BU_LABELS` and `BU_COLORS` of `panels.py`.

## Work in progress (as of 2026-10-24: S15 of the fast regime in the user's layout, good for now)

Goal: a manuscript figure showing that apical top-down input can support bottom-up (basal) plasticity that speeds
learning beyond the apical loop alone. The exploration was done in `sandbox/` (see `sandbox/README.md`); the
chronological notes of everything tried, including the rules that were set aside, are in `sandbox/NOTES.md`.

### Port into the main code (done 2026-10-13 and 14; the apical rule on 2026-10-22)

- control4 is `DEFAULT` / `APICAL_INHIBITION` (it produces all model figures, Fig. 4 and S11 to S13);
  `MIXED_SELECTIVITY` and `BU_PLASTICITY` are gone; `BU_HEBB`, `BU_BCM` and `BU_CALCIUM` carry hebb4, bcm4 and
  calcium4 (biological regime) and hebb3, bcm3 and calcium3 (fast regime); fast learning and (for the `BU_*`
  members) apical inhibition are options of `simulate_seeds`, `get_params` and `get_path`. Decisions at the start of
  the port (user): slim pickles, and the Smith estimator ported into the package (no Matlab on this machine). On
  2026-10-14 (user) everything Matlab went and the estimator moved into `helper.py`; each pickle now gets its
  performance when it is written. The regenerated results (`python main.py simulate`, 34 min on 8 cores) are again
  bit-identical to the sandbox figure sets and to the earlier sweep files.
- Verified: the 16 runs are bit-identical to the sandbox figure sets on every key, Smith traces and trials included
  (`w_bas` differs by design: initial weights here, weights after trial 0 in the figure sets).
  `python -m sandbox.check_reproduces` compares the ported loop with the mirror at the 8 settings x 2 protocols on
  short runs (identical). The sandbox keeps the old main-code constants (`LR`, `THETA`, `N_Z` 203, `THETA_0` 4) as
  its defaults, pinned in `sandbox/simulate.py`.
- Smith expert trials (seeds 0-9, mean +- sd, no inhibition / after the lift; every seed expert, no divergence):

  | regime | control | hebb | bcm | calcium |
  |---|---|---|---|---|
  | biological | 1100 +- 259 / 1155 +- 199 | 406 +- 41 / 330 +- 38 | 458 +- 42 / 494 +- 41 | 438 +- 80 / 330 +- 85 |
  | fast | 230 +- 56 / 267 +- 85 | 136 +- 23 / 85 +- 24 | 148 +- 26 / 104 +- 22 | 213 +- 42 / 168 +- 38 |

  control4 reads 1155.2 after the lift instead of the scan's 1154: the scans ran without the neuron relabeling, which
  changes the floating-point summation order and moves 5 seeds of the inhibition protocol by 1 to 4 trials.
- Final apical rule (user, 2026-10-22: port it as an additional rule for now): `BU_APICAL` (rule name `apical`,
  `results_apical.pickle` and `results_apical_perturbed.pickle`), dw = lr_bas (x_in - x_bar) (gain - 1) x_bas^2 with
  `l1`, `lr_bas` 0.005 and `xbar_tau` 50 (see Figure review), biological regime only (`get_bu_simulations()`;
  `get_runs()` has 18 runs, `get_hyperparameters()` refuses it in the fast regime). `basal_rule()` takes the basal drive
  before the gain as a new argument (the background at the outcome step). Verified: the two runs (44 s and 122 s on 8
  cores) are bit-identical to the figure set `apical4` on every key, Smith traces and trials included (`w_bas` as
  above); `check_reproduces` passes at 9 settings x 2 protocols (`apical4` added); seed 0 of the six biological rule
  runs and of fast hebb with inhibition re-simulate identically to the saved results. Smith expert trials (seeds 0-9):
  985 +- 205 / 1042 +- 199, every seed expert, faster than the control in 10 of 10 paired seeds in both protocols
  (signed-rank p 0.002), while fs14c's rank-sum test calls it n.s. (p 0.23 in both). The S14 draft panels show it as a
  fifth condition ('Apical', `#1baf7a`), pixel-identical to those on the final-rule page.
- `theta_0` scan at the biological rates (10 seeds, 1800 trials; Smith expert trial, seeds not expert, fraction of
  neurons with final `w_ap` above 0.5, selective neurons in the first / last 100 trials): 1.0: 1276, 2 lost, 1.00, 13
  / 67 percent; 1.5: 1179, 2, 0.88, 13 / 60; 2.0: 1024, 1, 0.41, 13 / 34; 2.35: 1100, 0, 0.25, 13 / 23; 2.5: 1111, 0,
  0.21, 13 / 20; 3.0: 1182, 1, 0.12, 13 / 15; 3.5: 1373, 1, 0.07; 4.0: 1532, 4, 0.04; 4.5: 9 of 10 lost; 5 to 8: no
  seed learns (performance 0.50, `w_ap` decays to 0 in every bin). Convergence of the panel's values (largest change
  of a bin mean over the last 300 trials): at most 0.04 up to 3.5; the most selective bin still rises by 0.07 at 4,
  0.11 at 4.5 and 0.06 at 5. User decision (2026-10-14), made on the decision page
  https://claude.ai/artifact/Py7sEPCCBMokPZnstY2CYm (the panel for three grids, the convergence chart and this
  table): the grid is 1 to 5 in steps of 0.5 plus 2.35 (`get_thetas_0()`, 10 files in `results/theta_0/`). The
  panel PNGs of the options and the convergence chart are in `panels/theta0_options/` (gitignored).
- Panels: the S12 panels read `DEFAULT`; fs12cd has the range -10 to 10 with values clipped into it (the pie counts
  every neuron) and the legend in the lower right; fs12c has its axis at +-`MAX_GAIN`; fs12b_theta0 has integer
  ticks. Every panel plots from the new results.

### Figure review (started 2026-10-16)

- User: finalize the figures first, then slim the results (plan in Next).
- f4g (and S13, drawn by the same function): the sensory dendrites are now averaged over the neurons that are
  texture-responsive in the first 100 trials (417 per seed, 330 to 532: 61 go-, 67 no-go-selective, 290
  non-selective; 47 percent reach `w_ap` > 0.5 against 11 percent of the others), instead of the go and no-go neuron
  (neurons 2 and 1, the first of 2 to 9 neurons tied at the clipped texture weight 1; both end at `w_ap` = 1). User
  decision (2026-10-16): the responsive neurons, meant to match a random sample of L5 neurons in barrel cortex, and
  the trace axes of f4g limited at 0.9 with ticks at 0, 0.4 and 0.8 (`y_lim`, `y_ticks` of
  `plot_apical_trace_boxplot()`; S13 keeps the limit at 1, its outcome dendrites reach 0.81 in Hit and 0.96 in FA
  trials). Same shape in every selection, smaller amplitude as it widens (touch naive to expert: go and no-go neuron
  0.31 to 0.91, responsive 0.17 to 0.47, all neurons 0.14 to 0.29; 75 percent of all neurons end with `w_ap` at or
  below 0.5 and their sensory dendrites stay at rest). Comparison page of the three:
  https://claude.ai/artifact/DMFf5tdoR4zvCTj27FTJ7X (PNGs in `panels/f4g_options/`, gitignored).
- Fixed: the expert-aligned traces of f4g and S13 drew the 936 trials every seed has around its expert trial (841
  before, 95 after) on the -500 to +100 axis, so the dashed line sat about 150 trials before the expert trial (also in
  the original code; right only for exactly 500 before and 150 after). f4f's legend had the go and no-go neuron
  swapped since the first commit (neuron 2, `T1_IDX`, is the go neuron). The boxplots of f4g use only 3 seeds in the
  learning phase (the other 7 have the Smith learning trial equal to the expert trial); the user does not mind.
- Cleanups (2026-10-16, verified): f4f's unused `x_pre_t` raster code is gone (the raster panel was dropped in
  46fb15e); the list branch of `get_texture_specific()` tested arrays with `in` and is fixed; the eligibility trace of
  `simulate_seed()` is a per-trial vector instead of a trials x neurons array (`check_reproduces` 16/16 identical, seed
  0 of the default run and of hebb with inhibition identical to the saved results on every key). f4f, S12c, S12d and
  S13 are pixel-identical to before.
- f4e (apical and soma rasters), S11e (the same apical raster function) and f4f (user, 2026-10-16): the single go and
  no-go neuron are replaced by the initially go- and no-go-selective neurons (61 and 67 per seed, the S12d counts; the
  single neurons are in their group in every seed). The apical side barely changes (last 300 trials: `w_ap` 0.94 and
  0.93 against 1.00, texture-time gain 8.6 against 9.1, sensory dendrite at touch 0.86 against 0.91); the soma response
  in the texture window is about half (4.6 against 9.4; the single neurons' texture weight is clipped at 1), shown with
  the soma color scale up to 5 (user; was 10). f4f (user): the cue line is the initially cue-selective neurons too
  (`get_cue_selective()`: 58 per seed, 46 to 74, the old cue neuron included in every seed, 2.6 per seed also go- or
  no-go-selective; final `w_ap` 0.91, cue-time gain 7.5, against 1.00 and 8.2 for the single neuron; merely
  cue-responsive would be 309 per seed, mostly texture-responsive too), and "Other neurons" is every remaining neuron.
  Each group's gain is taken while its stimulus is present (`get_gains()` at the cue or the texture step): the recorded
  gain is taken at each neuron's strongest initial input, the texture for only 44 to 46 percent of the texture-selective
  neurons, which would give 4.5 instead of 8.6. Legend labels "Cue-selective", "Go-selective", "NoGo-selective" (longer
  ones do not fit the panel). The soma figures keep their row labels (go group on top, no-go group below, as for the
  single neurons). Before/after page: https://claude.ai/artifact/MEwDJUGGvdCXhK8YjFPCKr (PNGs in `panels/f4ef_options/`,
  gitignored).
- New supplementary figure on the rules, biological regime (draft, 2026-10-16; "S14" in the names is a placeholder):
  `fs14a` / `fs14b` mean +- sd performance without / with apical inhibition (legend in b, where every condition is at
  chance), `fs14c` expert trials unperturbed and corrected (rank-sum against the control of the same protocol next to
  the bars, unperturbed against corrected in brackets), `fs14d` the basal weight profile (each neuron's 33 weights at
  trial 1800 sorted, rank 1 = largest, mean weight per rank over all neurons and seeds; the control line is every
  network's initial profile), `fs14e` the percentage of texture detectors (neurons with a weight above
  `DETECTOR_THRESHOLD` 0.5 from either texture, from the snapshots every 50 trials), solid without and dashed with
  inhibition. The first draft's "strongest input is a texture" depended on the input order (`argmax` ties: the rules
  leave 36, 7 and 48 percent of the neurons with two or more inputs tied at their largest weight, mostly at 1) and was
  replaced the same day after the user asked for the definitions of d and e; `fs14f` the S12d classes in the first and
  last 100 trials (stacked bars), `fs14g` the S12d histograms of the last 100 trials for the four conditions, `fs14h`
  naive (before the learning trial) and expert outcome dendrites and sensory dendrites at touch (neurons
  texture-responsive in the first 100 trials, as f4g). Colors: control grey, hebb `#2a78d6`, bcm `#eb6834`, calcium
  `#4a3aa7` (validated for color vision deficiencies). Numbers (control, hebb, bcm, calcium): texture detectors 4.5,
  9.3, 9.6, 11.0 percent at trial 1800 and 4.5, 8.3, 5.9, 10.7 at the end of the inhibition (so only bcm's sharpening
  needs the apical signal in these runs); strongest input 0.66, 0.89, 0.74, 0.80; inputs above 0.05 per neuron 7.4, 1.5,
  1.9, 2.1 (hebb and calcium leave 11 and 18 percent of the neurons without input); unresponsive in the last 100 trials
  52.7, 82.1, 86.1, 60.8 percent, strongly selective (beyond +-5) 3.5, 8.5, 9.1, 10.7; outcome dendrites naive to expert
  0.78 to 0.26, 0.79 to 0.25, 0.84 to 0.26, 0.76 to 0.29, sensory dendrites at touch 0.17 to 0.47, 0.11 to 0.60, 0.11 to
  0.23, 0.16 to 0.46. Since 2026-10-22 the apical rule (`BU_APICAL`) is a fifth condition (label 'Apical', color
  `#1baf7a`; with the three rule colors it passes the color-vision check, worst pair 9.2, and has 2.7:1 contrast on
  white, so it needs its labels): texture detectors 17.1 percent at trial 1800 and 4.5 at the end of the inhibition (no
  weight change while inhibited), tone detectors 9.3 (control 2.2), strongest input 0.75, inputs above 0.05 per neuron
  7.3 (none left without input), unresponsive 50.3 percent, strongly selective 16.9 (nearly all of its 18.7 percent
  S12d-selective neurons; control 3.5 of 22.6), outcome dendrites 0.82 to 0.30, sensory dendrites at touch 0.13 to 0.46.
  Review page with the layout, the numbers, a draft legend and a fast-regime preview:
  https://claude.ai/artifact/1QCh5kWyQZQc8wikz1JdGD. Open for the user: d as the rank profile or as the number of inputs
  above 0.5 per neuron (alternative on the page), keep h, split e by protocol, labels ("Control", "Hebb", "BCM",
  "Calcium"), numbering, additions (a rule schematic, tone detectors, a `rule_sees_gain` ablation at the current
  settings). `fs12cd` now uses the shared helpers; S12d, 4e, 4f and S11e render pixel-identical to the committed code.
- Why `fs14e`'s dashed lines rise during the inhibition (user, 2026-10-16; checked in the data): the apical gain is
  exactly 1 at every step and `w_ap` never changes in the 30 inhibited runs, so the rules run on the basal drive alone.
  hebb: fixed `theta_bas` (median 0.38); a texture synapse goes to 1 when its weight plus `BKG` exceeds it, which
  predicts 73 of the 83 detectors per seed from the initial weights (99 percent of neurons alike; 86 with the average
  coincident distractors). calcium: the `pre` bound keeps each stimulus's total weight over the neurons (55 and 58 per
  texture) while the weights go binary (0.6 percent of texture weights end between 0.05 and 0.95), so each texture ends
  on about as many neurons at weight 1 as its total (54 and 56, 10.7 percent), with or without apical activity. bcm:
  `theta_m` starts at the apical threshold and rises about fourfold within 150 trials, so the early detectors (7.7
  percent at trial 150, seed 0) lose the race against it for 32 of 85 neurons; with apical activity half of those keep
  the texture synapse (9.6 against 5.9 percent at trial 1800). Instrumented re-runs of seed 0 (wrapping `basal_rule`,
  bit-identical to the saved runs) and the explanatory figure are on the review page. Performance stays at 0.51 because
  the silenced outcome activity stops the value estimate's learning (its update has the factor `x_som - BKG` = 0 at gain
  1) and leaves the policy update at a tenth.
- Equations of the three rules (user, 2026-10-17): on the review page with the constants, each rule's postsynaptic
  factor drawn for the median neuron, and LaTeX for the methods. A float64 re-implementation written from the equations
  reproduces `basal_rule`, `bound_basal_weights` and the initial thresholds to float32 precision (both regimes and
  protocols, random states with apical activity). calcium's level `c = x_som (1 + ca_beta a)` reads the neuron's own
  apical activation (`a = (gain - 1) / (MAX_GAIN - 1)`), so for calcium "no explicit apical factor" only means "not
  gated" (the page and the draft legend now say "none of the rules is gated by the apical activity"). `ca_beta` 0
  changes nothing at calcium4 (seeds 0-9: 440 against 438 trials, p 0.79; 330 against 330 after the lift; texture
  detectors 10.9 against 11.0 percent): without the term the strongest input already reaches `theta_p` at gain 1.5.
  bcm's threshold relaxes from theta to theta / `bcm_e0_scale` = 4 theta at the initial weights and gain 1 (the code
  comment saying theta was its fixed point is corrected, as is the calcium docstring's "least depression above
  theta_p": Omega is -0.13 at zero calcium, -1.67 at its minimum at 0.77 theta_p, -1 above 1.7 theta_p). The soft
  bound makes 1 absorbing (3.4 and 1.1 percent of the hebb and bcm weights are exactly 1 at trial 1800, 0.5 initially);
  calcium's `pre` leaves 93.5 percent exactly 0. Open for the user: keep `ca_beta` 2 and describe the apical term, or
  set it to 0 (re-simulating calcium4 and calcium3) so that every rule sees the apical signal only through `x_som`.
- No plasticity at gain 1 (the user's intuition, 2026-10-17): the share of the weight change made while the neuron's
  gain is below 1.1 is 95 (hebb), 88 (bcm) and 84 percent (calcium; instrumented re-runs of seeds 0-2, computation
  unchanged), about 90 percent of all of it on distractor synapses (the pruning; 63 percent depression for hebb and bcm,
  all of it for calcium); multiplying each update by `a = (gain - 1) / 9` would leave 4.3, 6.2 and 7.4 percent. The
  apical signal sits only at the tone and texture steps (control, last 300 trials: `x_pre_t` 0.46 and 0.52, 0.007 at the
  other 15 steps, mean gain 1.01 there) and is the same at both (gain 2.9 and 3.1), so a gated rule acts only there, on
  the 14 to 27 percent of neurons with apical weight, and late (at touch 4 to 10 percent of the neurons reach gain 1.1
  in the first 300 trials). September's gated rules (`gated_hebb`, `burst`, `us` in `sandbox/rules.py`, older 203-neuron
  model, see `sandbox/NOTES.md`; formulas and reasons recapped for the user on 2026-10-18) were at most 3 to 7 percent
  faster than the control and diverged via the tone. Only `gated_hebb` is exactly silent at gain 1: `us`
  (`x_bas (gain - g_bar)`) depresses at gain-1 steps while its running-average gain is above 1 (about 1.4 early, lifted
  by the outcome step), and `burst` (`x_som (P - P_bar)`, `P = x_ap + burst_kappa x_som`) does the same and, at its
  finalists' `burst_kappa` 0.2, stays active with the apical input silenced. Options given to the user: an explicit gate
  (`a` times each rule's factor, `gated_hebb` for hebb; calcium with `c = ca_beta a x_som` and Omega(0) = 0), thresholds
  out of reach at gain 1 (calcium: `ca_ltd_frac` about 0.8 to 0.9 or a larger `ca_theta_p`, `CA_SMOOTH` about 0.02; hebb
  and bcm need a third, no-change threshold as in Artola-Broecher-Singer), or gating only potentiation (keeps the
  pruning; detectors form only with apical activity). Calcium also needs a per-neuron bound (`pre` moves the weights of
  neurons at gain 1) and then `ca_gamma_d` below 1. All three were tried in some form in the sandbox (older model): the
  gate as `gated_hebb` (above); potentiation-only gating in effect as `us` and `burst` plus `hebb`'s depression half
  (`lr_dep`, `thr`), which were identical to that depression alone (896 and 885 against 895 trials; their gated
  potentiation is too weak at every rate that does not diverge); and the thresholds for calcium as a narrow LTD window
  (sweep `s2_casilent`, 2026-09-29: `ca_ltd_frac` 0.8 to 1.0, `l1`, `ca_gamma_d` 0.5, 5 seeds; summary in git at
  6d09ed0), which at best matched the control (603 against 597 trials at `init_power` 12, 891 against 814 at 10), with
  no seed learning in 53 of its 54 settings with `ca_beta` 2. So in the old model every variant tried that is silent (or
  nearly) at gain 1 learned at about the control's speed. Awaiting the user's choice of variant to test at the current
  settings.
- Gated rules re-tuned on control4 (user, 2026-10-18; sandbox only, details in `sandbox/NOTES.md`, results page
  https://claude.ai/artifact/U1wK8J511MzycH2yrYmacp): `gated_hebb`, `burst` and `us` with control4's rates fixed and
  only their own bound, `lr_bas` and constants free, the running averages starting at rest (user) and `burst` reading
  the apical activation rescaled to 0 at rest (the inhibited outcome step's `x_ap` of 0 had made it potentiate during
  inhibition). Smith expert trials over seeds 0-39, no inhibition / after the lift, every seed expert: control 1122 /
  1163; `gated_hebb` (`l1`, `gh_post` `bas`, `lr_bas` 0.512, `theta_bas_0` 14) 657 / 638 at `gate_power` 1, 434 / 423 at
  0.5, 403 / 385 at 0.25 (rank-sum against the control p below 1e-12); `us` (`l1`, `us_tau` 5, `lr_bas` 0.0005) 1020 /
  1044 (p 0.048 / 0.01); `burst` (soft, `burst_kappa` 0, `burst_tau` 20, `lr_bas` 0.002) 1062 / 1082 (p 0.23 / 0.056;
  `burst_kappa` 0.1 changes nothing). `gated_hebb` needs the un-gained drive (`som` diverges) and `l1` (soft at best
  913, `pre` diverges). The gate power trades speed for plasticity near rest: 10, 30 and 57 percent of the change at
  gain < 1.1 at `gate_power` 1, 0.5 and 0.25 (hebb4 95, bcm4 88, calcium4 84); outside inhibition the gain is almost
  never exactly 1, so a small power approaches an ungated rule. All three are exactly silent during inhibition and learn
  about as fast after the lift as without it (hebb4 and calcium4 gain from the blind trials: 330). `us` and `burst`
  depress at gain 1 because the outcome step lifts their running averages above rest (as in September). Open for the
  user: which gate power counts as gated, whether gated Hebb enters the main code (a fourth rule or a replacement), and
  whether to commit `sandbox/results/gated4/` (4.5 MB).
- Denoising or apical selection (user, 2026-10-19; page https://claude.ai/artifact/TEQc33iyRMba3qj8rfNQQM, details in
  `sandbox/NOTES.md`): hiding the gain from the rule (`rule_sees_gain='never'`, the network keeps it) leaves hebb4,
  bcm4 and calcium4 as fast as before (410, 461, 439 against 406, 458, 438 trials, seeds 0-9), and with the gain hidden
  a lower threshold is faster still (hebb `theta_bas_0` 5: 301; calcium `ca_theta_p` 1.0: 388); gated Hebb without
  its gate learns in 389 (`lr_bas` 0.128). The apical gain appears only as the network learns (under 3 percent of the
  neurons above gain 1.1 at touch in the first 200 trials) and carries 0.7, 13.7 and 2.8 percent of the rules' weight
  change up to their expert trial; the expert trial follows the inputs left per neuron at trial 300 (r 0.91 over 16
  variants). `us` and `burst` amplify whatever is active at cue and touch, and the apical input is the same for both
  textures (one `x_pre_t` per time step), so they add unselective texture and tone detectors but no speed; freezing
  the tone (`rule_frozen_inputs`) stops their failures at moderate rates, not their slowness. Open for the user:
  whether the figure claims apical support of the bottom-up plasticity (that needs an apical signal present early or
  different between go and no-go) or presents the rules as apical-independent denoising.
- Apical-deviation rule (user, 2026-10-19; page https://claude.ai/artifact/UuEfGsvBYtXegFdyGShyqz, details in
  `sandbox/NOTES.md`): `apdev`, dw_k = lr x_k (g - g_bar) (the x_ap version at 10 lr; silent during inhibition), best
  1022 trials over seeds 0-9 (soft bound, `apdev_tau` 20, `lr_bas` 0.001; control 1100, unpaired p 0.5; paired by
  seed faster in 36 of 40 seeds by 99 trials, p 1e-7, see the bullet on the rule without g_bar); every other bound
  fails. The outcome step lifts g_bar (1.44 by trial 80), so before the cue and touch signal the rule depresses by
  activity (the tone most); then the apical input, the same for both textures, raises both together (contrast unchanged,
  both-texture neurons) and the tone. The control started with its texture weights x2, x4, x8 learns in 504, 286,
  205 trials, so the user's target state is worth a lot. Open for the user: texture identity in the apical input, a
  basal factor (gated Hebb, 641 at `gate_power` 1) or a baseline the outcome does not dominate.
- Faster apical-deviation rule (user, 2026-10-19; page https://claude.ai/artifact/5zmyn78X7HjMjvGcppuyHG, details in
  `sandbox/NOTES.md`): faster is worse. From trial 1, `lr_bas` 0.0005 to 0.004 give 1050 to 1183 trials, 0.008 3 of
  10 seeds, 0.016 to 0.128 at most 1; switched on at trial 500 (`rule_start_trial`), 0.001 1046, 0.002 4 of 10, more
  at most 1 (tone frozen too: 953 at 0.002, at most 1 of 10 above). Before the touch signal (no neuron above g_bar
  before trial 500) the rule only depresses, which keeps neurons from being recruited (touch gain above g_bar at
  trial 1000 in 21, 6, 1, 0 percent at 0.001 to 0.032); where it potentiates it raises the tone and both textures
  together. The control started with both textures raised by the same amount (`init_task_mode` `'shift'`, each
  neuron's w_T1 - w_T2 kept) learns slower or never (x2 1240, 7 of 10; x4 and x8 no seed, licking on every trial;
  ratio kept: 504, 286, 205): the policy update has no baseline (hit +1, false alarm -0.5), so activity shared by go
  and no-go trials drives licking. The earlier page's "premise holds" is corrected there. Open for the user: as for
  the apical-deviation rule.
- Apical rule without g_bar, with normalization (user, 2026-10-19; page
  https://claude.ai/artifact/KcNfjUBvFmBdUUXMH2htJt, details in `sandbox/NOTES.md`): `apdev_bar` `False`, dw = lr x g.
  Additive fails under every normalization (at rest it pulls each neuron's weights towards the input frequencies).
  Proportional to the weight with `pre` (exactly silent at gain 1) or `both`: no depression; once the touch signal
  appears (trial 600 to 900) all texture and tone weight moves onto the neurons with apical weight (textures 1.8-1.9
  times there, 0 elsewhere, preference kept). `pre`, `lr_bas` 0.003: 1011 against 1122 on seeds 0-39, faster in 40 of
  40 seeds by 111 trials (10 percent; `both` 104); higher rates add nothing. It cannot move the touch-signal onset
  (720 against 708), only shorten onset to expert (380 to 300 trials). Comparisons are paired by seed from now on
  (same initial weights and stimuli). The user's dw = lr x (g - 1) (rest subtracted): additive it is slower than the
  control in every seed and loses seeds under every normalization (both textures get the same increment, preference
  0.77 to 0.16-0.42); times the weight with `pre` it is faster in 39 of 40 seeds by 111 trials (the same rule as the
  no-baseline x w one). Slower additive rates (user, 2026-10-20) only approach the control (`pre` 3e-6, 1e-6, 3e-7:
  11, 2 and 0 trials slower, never faster); at 1e-4 the recruited neurons' tone, T1 and T2 weights converge
  (preference 0.77 to 0.16, response go/no-go contrast 0.41 to 0.14, selective neurons 22.5 to 12.9 percent), so
  selectivity drops but does not vanish. The x w rule keeps the preference but leaves 84 percent of the neurons
  unresponsive to the textures (control 53). Times the basal drive instead of the weight (user, 2026-10-20;
  `apdev_mult` `'bas'`): `pre` at most 13 trials faster (0.0001: 7, 32 of 40 seeds; 0.0003: 13, one seed lost),
  `both` and `l1` slower, every bound fails from 0.003 or 0.01; x_bas has a floor (background and coincident
  distractors, about 0.05), so the weak texture's update is nearly additive and the preference erodes; the soft
  bound (which brings the weight back) at 0.001 is 34 trials faster in 39 of 40 seeds with the control's population.
  Why x_bas fails under `l1` (user, 2026-10-21; instrumented runs, ablations, page v5): every synapse active at a step
  gets the same increment, so 51 percent goes to coincident distractors and the tone (every trial, cue gain = touch
  gain late) doubles; the L1 rescaling takes in proportion to the weight, so the preferred and the other texture net
  +0.046 and +0.053 (gap flat, preference 0.80 to 0.55); never faster at 0.0001 to 0.001, and a fresh control started
  from its trial-1800 weights never becomes expert (always licks). Textures only: 10 of 10 faster. Times the weight
  and the basal drive (`apdev_mult` `'wbas'`, dw = lr x w x_bas (g - 1)): the preference sharpens; seeds 0-39 `pre`
  0.003 964 against 1122 (40 of 40, 157 trials, 14 percent; faster than the x w rule in 37 of 40 by 47), `l1` 0.003
  994 (39 of 40, 127). (x - x_bar) and x_bas^2 (user, 2026-10-21; `apdev_xbar`, `apdev_bas_power`; page v6, details in
  `sandbox/NOTES.md`): x_bas^2 alone is at most 27 trials faster (the floor's variance gives the weaker texture as much
  per unit weight). With x_bar the input expected at that step of the trial ('slot') the tone is never potentiated and
  the textures push against each other (preference 1.00); seeds 0-39, `l1`: x_bas^2 0.03 902 against 1122 (40 of 40, 20
  percent), times the weight as well (dw = lr (x - x_bar) w (g - 1) x_bas) 0.1 786 (40 of 40, 30 percent; faster than x
  w x x_bas in 40 of 40), but no recruited cue neurons are left (`pre` keeps them, 839 to 950 on seeds 0-9, distractors
  pile up). A time average ('step', x_bas^2) gives 989, the local apical-event average ('apical') 1001 (the tone grows
  on the cue neurons; with the tone frozen 883 and 951 on seeds 0-9). The user set the trial-locked x_bar aside as not
  biological and prefers the time average (2026-10-21, details in `sandbox/NOTES.md`, page v7): with it x_bas linear
  erodes the preference (seeds 0-39 1075, 39 of 40 faster by 47 trials) and a power from about 1.5 is needed (1004 / 989
  / 982 at p = 1.5 / 2 / 3; the expected drift of ln(w_P / w_O) crosses 0 near p 1.45). The weight does the same job: dw
  = lr (x - x_bar) w (g - 1) x_bas under `pre` learns in 911 / 864 / 776 / 640 / 507 / 432 trials at 0.01 / 0.03 / 0.1 /
  0.3 / 1 / 3 (40 of 40 faster at each rate; without x_bar `pre` loses seeds from 0.01 as distractors pile up on the
  recruited neurons), is exactly silent at gain 1 (no weight change during inhibition, after the lift 641 / 448 at 0.3 /
  3) and moves the touch-signal onset (630 / 450 / 295 at 0.03 / 0.3 / 3, against 715); costs: distractor build-up at
  0.01-0.1, 1-2 inputs per recruited neuron from 0.3, 28 to 51 of the 52 cue-selective neurons left; `l1` 986 at 0.01
  (as x_bas^2). The user chose the squared time-average rule under `l1` as the final version (next bullet).
- Final apical rule (user, 2026-10-21: the squared version with the time average, without `pre`; page
  https://claude.ai/artifact/4ACjibSDakeKpWts2xh7Py, details in `sandbox/NOTES.md`): dw = lr (x - x_bar) (g - 1)
  x_bas^2, x_bar each input's running average over all steps of the trial (outcome step included, tau 50 trials),
  per-neuron `l1`, lr 0.005 (sandbox: `apdev`, `apdev_tau` inf, `apdev_mult` 'bas', `apdev_xbar` 'step',
  `apdev_bas_power` 2). Seeds 0-39: 988 against 1122 without inhibition (39 of 40 faster, 12 percent), 1009 against 1163
  after the lift (40 of 40, 13 percent); no weight change during inhibition, with the gain hidden it is exactly the
  control; acts only at the cue and the touch once the touch signal exists (onset unchanged, 0.5 percent of its change
  at gain < 1.1); scan on seeds 0-9: every lr from 0.002 to 0.02 faster in 10 of 10, tau 10 to 200 the same, the soft
  bound and no normalization fail. Recruited neurons: preference 0.79 to 0.98, distractors 1.42 to 0.89, but the tone
  grows (0.14 to 0.34; tone detectors 2.3 to 9.5 percent, cue-selective by weights 52 to 97); texture detectors 17
  percent (control 4.4, hebb4 to calcium4 9 to 11); S12d selective neurons in the last 100 trials 19.1 against 22.3
  percent (not analysed; seeds 0-9: 18.7 against 22.6, of which 16.9 against 3.5 strongly selective, beyond +-5); fresh
  control from its end weights 236 trials. Figure set `sandbox/results/figures/apical4` (S14 panels with the rule as a
  fifth condition, since 2026-10-22 drawn by the main code; the draft's rank-sum test on seeds 0-9 calls it n.s.). Fast
  regime (user, 2026-10-22: optimize it there; page https://claude.ai/artifact/EQMVLQ6ASuPuv9DyTUsYX2, seeds 0-9): no
  setting works at control3. Every bound (`l1`, `pre`, `both`, `soft`, `clamp`) loses seeds at every rate from 1e-4 by
  divergence of the TD value estimate, and the stable rates (`l1` 1e-5, 3e-5) change nothing; x_bar's tau 10 / 200 and
  power 1 / 3 do not help. Mechanism: in no-lick trials the end-of-trial value correction scales the value weights by
  about 1 - lr_td K_V S (S = sum_t |x_t|^2 + sum_t x_t . z_t); control3 peaks at lr_td K_V S = 1.90 against the
  overshoot limit 2, and the rule triples S. With a slower fast control it is stable but small: TD 0.0025 no faster,
  0.002 318-322 against 339 (8 of 10 faster), 0.0015 393 against 406; the touch signal appears only 60 to 116 trials
  before the expert trial (control3: onset 170, expert 230). TD 0.002 on seeds 0-39: η 0.01 faster in 35 of 40 seeds
  (median 16 trials) but slower on average (354 against 343: seeds 17 and 33 learn about 680 trials later), η 0.03 loses
  2 seeds; after the lift (seeds 0-9) η 0.01 337 against 359 (10 of 10). Recommendation on the page: leave the rule out
  of the fast figure (redefining the fast control would also mean re-running hebb3, bcm3 and calcium3). Ported on
  2026-10-22 as `BU_APICAL`, an additional rule (user; see Port). Open for the user: the fast figure without it, and the
  change of the cue population (the S14 composition and its paired test were settled on 2026-10-23, see the S14 bullets
  below).
- Panels of the apical rule (user, 2026-10-22: "first add any panel with the new bottom-up plasticity"; a to h already
  had it as the fifth condition): `fs14i` expert trials paired by seed, control against the rule, both protocols
  (signed-rank: 10 of 10 faster in both, 115 / 113 trials, p 0.002, the floor at 10 seeds; rank-sum in c p 0.23);
  `fs14j` summed basal weight change per neuron from trial 0 to 1800 of the recruited neurons (final `w_ap` > 0.5) and
  of the others, for the four rules (apical 2.27 against 0.021; hebb 2.77 / 1.59, bcm 1.93 / 1.23, calcium 2.38 / 1.80;
  final `w_ap` bimodal, 0.1 to 0.6 percent between 0.1 and 0.9); `fs14k` the recruited neurons' tone, preferred texture
  (larger initial weight, never tied), other texture and distractor sum over trials with the rule (0.14 to 0.34, 0.27 to
  0.61, 0.026 to 0.012, 1.42 to 0.89; the preferred texture rises from trial 800, after the touch signal at 705; the
  other 729 neurons per seed unchanged); `fs14l` the mean change over trials in both protocols (hebb, bcm, calcium reach
  83, 36, 64 percent of their trial-1800 change by trial 100 and change as much under inhibition; the rule changes
  nothing during inhibition, exactly 0 in every seed). Review page with the twelve panels, legend drafts, a worked
  example and the decisions (which of i to l, the test of c, the group, one figure or two; settled on 2026-10-23, next
  bullet): https://claude.ai/artifact/5yY1DiSrj24ne9rbojx94h. `helper.py` got `TONE_IDX` and `N_TASK_STIMULI` (panels
  only).
- S14 on the apical rule only (user, 2026-10-23: "focus fully on 'apical' ... the others should not appear ... no need
  to delete the code and results yet"): a the schematic (empty), b the performance panel of fig. 4h with the rule
  (`fs14b_performance`; expert trials 985 +- 205 unperturbed, 1042 +- 199 corrected, 2842 +- 199 perturbed; the rule's
  versions of the expert-trial and `w_ap` panels of fig. 4h were dropped the same day, user: rank-sum n.s. p 0.36
  corrected, p 1.6e-4 perturbed; the selective neurons' `w_ap` ending at 0.94, the unresponsive ones' at 0.13 against
  0.11, from the tone's growth), c the former i, d the former k (with a classic legend since the same day, user: two
  rows of two in the margin above the axes, right-aligned with them as in f4h_w_ap, the rows reading tone, preferred
  texture, other texture, distractors; the y-axis to 1.7 with its line ending at 1.5, which keeps the legend clear of
  the tick label, and the panel's top-left corner free for the letter), e fig. S12c with the rule (outer bins of initial
  basal selectivity end at -8.5 / +8.5, control -6.0 / +5.9; the +-0.2 bins at -5.7 / +5.4, control -1.3 / +1.3), f the
  after-learning histogram of fig. S12c/d with the rule (go 9.2, no-go 9.6, non-selective 31.0, unresponsive 50.3
  percent, control 10.9, 11.6, 24.8, 52.7; beyond +-5: 16.9 against 3.5 percent; the axis to 20 percent because the zero
  bin holds 16.9, the legend in the upper right, the histogram as wide as S12's, 1.12 `PANEL_WIDTH`). Fig. 4h, S12c and
  S12c/d were refactored onto the shared helpers and render pixel-identically. Page (version 3 of the S14 page, with b
  reduced and d's legend): https://claude.ai/artifact/5yY1DiSrj24ne9rbojx94h. The user approved this version
  (2026-10-23: "i like the current version"); only a, the schematic of the rule, is still to be drawn. Name (user,
  2026-10-23): the panels call the rule "Plastic $w^\mathrm{bas}$" (`BU_LABELS`; was 'Apical'). Only c names it, on
  two lines under its ticks (on one line it overlaps "Control"), with the protocol labels 22 pt below the axis
  (`annotate`, which the layout places exactly; the line spacing of before, axes 82.4 pt high against 87.7, b to f 83
  to 88.5). e's colorbar reads $w^{bas}_{Go}(t_0)-w^{bas}_{NoGo}(t_0)$ (the bins are the initial weights; for the
  `BU_*` simulations, S12c unchanged), the rest of e pixel-identical. "Apical inhibition" (4h, S14b) names the protocol
  and stays; the code names (`BU_APICAL`, 'apical', the result files) are unchanged, and the draft PDFs in
  `panels/s14_rules_draft/` still say 'Apical'. Version 4 of the S14 page (same URL; the name in its legend drafts, c
  and e marked as updated, S12c's colorbar as an open point) has its source in
  `sandbox/results/scratch_2026-10-22_23/page/` (`apical_rule_panels.html` with `img3/` and `img4/`, v3 kept as
  `apical_rule_panels_v3.html`; the render, measurement and page-edit scripts in `rename/`).
- f4h_w_ap (user, 2026-10-23: "show the average w_ap evolution of the initially texture selective neurons as well as the
  initially texture unresponsive neurons"): instead of the single go and no-go neuron (neurons 1 and 2, all at `w_ap`
  1), one line per seed and group with the mean `w_ap` of the neurons texture-selective (go or no-go, pooled: 0.94 and
  0.93 at trial 1800; solid) or texture-unresponsive (dashed) in the first 100 trials of each protocol (127 and 583 per
  seed without inhibition, 125 and 598 with; nearly the same neurons). Selective 0.10, 0.06 at trial 300, 0.94 at 1800
  (crossing 0.5 between trials 636 and 859; 94 percent end above 0.5), unresponsive 0.10, 0.04, 0.11 (11 percent above
  0.5); with inhibition no change until the lift, then 0.95 and 0.11 by trial 4000. Per-seed lines rather than mean +-
  sd (the seeds rise at different trials, so the band dips below 0; user: "i like the version you chose"); a classic
  legend (user), "Texture-selective" and "Texture-unresponsive" with solid and dashed lines, in two rows in the margin
  above the axes, right-aligned with them (`fig.legend` after `adjust_figure`, so the axes keep the size of the other 4h
  panels; the traces leave no room inside and one row is wider than the panel). Open for the user: 46 of the 58
  cue-selective neurons per seed are texture-unresponsive and hold 69 percent of that group's final `w_ap` (0.04 without
  them). Page with before and after: https://claude.ai/artifact/PAtiGhH85NqU3krF3oqFCR (PNGs of every version in
  `panels/f4h_options/`, gitignored).
- Still to review with the user: f4c, f4d, the f4h performance and expert-trial panels, S11b to S11d, S12 and S13.
- Fast-learning regime (user, 2026-10-23: "figure out how or whether to show the fast parameter optimization
  simulations ... replot all the control results with the fast parameters ... feel free to add an other supplementary
  figure"): every panel of figs. 4 and S11 to S13 but S11b/c is redrawn from control3 into `panels/fast/` (same file
  names), and a draft S15 (placeholder name, `panels/fig_s15*`) compares the regimes: a mean performance, b per-seed
  expert trials fast against biological in both protocols (a seed has the same initial weights and stimulus sequence in
  both regimes, checked; proportional fit fast = biological / 4.5, Pearson r 0.92 / 0.88), c and d trials to expert and
  recruited neurons (final `w_ap` above 0.5) in the two `theta_0` sweeps, e the outcome dendrites in Hit trials over
  trials, f the reward prediction (V̂ at the outcome step) in the Hit trials of the last 300 trials. New for it: the
  fast `theta_0` sweep (1 to 2 diverge in every seed, at trials 87 to 613; 2.5 in 2 of 10, the other 8 expert at 213;
  then 230, 296, 391, 479 and 685 trials from 3 to 5, every seed expert), the divergence stop of the loop
  (`check_reproduces` 18/18 identical), `x_som_cue` in the fast default run, the f4g / S13 window rule, the Tukey skip
  and the shading of `fs12b_theta0` (see Architecture); the 27 biological and S14 panels render pixel-identical.
  Findings: expert after 230 / 267 trials (biological 1100 / 1155); apical inhibition still blocks learning
  (performance 0.53 at trial 1800; corrected against unperturbed p 0.38, perturbed p 1.6e-4); recruitment is set by
  `theta_0` (equal in both regimes at equal `theta_0`: 11.7 / 11.3 percent at 3, 3.7 / 4.2 at 4), so the fast regime's
  11.3 percent (25.2) follows from the `theta_0` 3 its TD rate needs; the initially texture-selective neurons end at
  `w_ap` 0.56 (0.94), 56 percent of them recruited (94), gains at their stimulus 5.1 to 5.4 (7.5 to 8.6); selective
  neurons 12.8 to 15.1 percent (12.7 to 22.5); every fast seed learns even at `theta_0` 5 (685 trials with 1.4 percent
  recruited; biological none); the Hit-trial outcome dendrites stay at 0.67 after learning (0.32) because V̂ at the
  outcome scatters from trial to trial (sd 0.55 against 0.16, mean |1 - V̂| 0.45 against 0.15, correlation between
  consecutive Hit trials -0.01 against 0.71; the TD rate sits at 1.9 of the overshoot limit 2, see the fast-regime
  analysis of the apical rule); sensory dendrites at touch 0.12 to 0.25 (0.17 to 0.47), touch-signal onset 60 trials
  before expert (385); the Smith learning trial is the expert trial in 9 of 10 fast seeds. Page with the replots side by
  side, S15 with draft legends, the numbers and the options: https://claude.ai/artifact/17KgLcqMJ6zCfnTHY1VXaZ (source,
  render and number scripts in `sandbox/results/scratch_2026-10-23_fast/`). Recommended there: one supplementary figure
  (S15 a to f, possibly with the fast 4h performance panel), the biological regime for every main figure, the replots
  as review material.
- S15 in the user's layout (2026-10-24: "a with the same as fig4h performance traces ... b like fig_s14c but with the
  fast parameters ... c something like the current panel b ... d, e, f ... showing that the parameters are at the edge
  of divergence ... fig 12c_theta_0 [the fast S12b theta_0 panel] together with the current panel c as d and e ... f
  the expert times depending on the other fitted parameters like the learning rates"): a fig. 4h's performance panel in
  the fast regime (with inhibition: performance 0.53 at trial 1800, expert 267 +- 85 trials after the lift); b fig.
  S14c's paired expert trials, biological against fast (10 of 10 seeds faster in both protocols, by 870 and 888 trials,
  signed-rank p 0.002); c the former b (`fs15c_speedup`); d fig. S12b's `theta_0` panel in the fast regime (at equal
  `theta_0` the fast bin curves follow the biological ones: top go bin 0.72 against 0.82 at 3, 0.35 against 0.37 at 4);
  e the former c; f new learning-rate sweeps of the fast regime (`results/fast/lr_*/`, 54 runs, 47 min; each rate
  alone, 1/8 to 8 times, `theta_0` and the others at their fast values; four axes in a row, two panels wide). Dropped: the
  former d (recruited neurons), e (Hit outcome dendrites) and f (reward prediction); their code is in
  `sandbox/results/scratch_2026-10-24_s15/before/panels.py` and their PDFs in `old_s15_files/` there. New in d, e, f
  and the fast S12b replot: a lighter band where some seeds diverge (`COL_PARTLY_DIVERGED` 0.955; the full band is now
  0.9, was 0.92). Sweep results (trials to expert, mean +- sd over expert seeds; seeds 0-9): TD 1047 +- 185 at 1/8 (the
  biological TD rate; 9 expert, seed 9 locks into licking on every trial; biological regime 1100), 817, 614, 466, 368,
  279, 248 at 0.84, 230 at 1; 1.19 223 (1 diverged, trial 293), 1.41 250 (4 diverged, trials 228 to 1462), 2 to 4 and
  8 every seed diverges, 5.66 one seed expert at 791 (9 diverged). So the TD rate alone accounts for nearly the whole
  speed difference between the regimes, and it is at the divergence edge (as `theta_0` is from below: 2.5 loses 2 of
  10). Policy: 307 at 1/8, 275 at 1/4 (the biological rate), 230 from 0.84 to 1.19, 292 at 2; from 2.83 some seeds
  never become expert, locked into licking on every trial or on none (1, 1, 3 and 10 of 10 at 2.83, 4, 5.66 and 8), no
  divergence. Apical weights: 230 to 236 from 0.5 to 2; below 0.5 1 or 2 seeds diverge (1 at 0.35 and 0.25, 2 at 0.18
  and 1/8; recruited 19.7 percent at 1/8 against 11.3), above 2 slower and fewer recruited (378 trials and 4.6 percent at
  8); the biological value (0.22) is in the diverging range of the fast regime. Apical input (`lr_trace`): 222 to 241
  from 0.5 to 3.9 (its maximum, 1.0), up to 302 when slower. Recruitment hardly moves with the TD (8.4 to 11.8 percent)
  and policy rates (10.1 to 11.3). Verified: `check_reproduces` 18/18 identical after the `lr` option; the 27 biological
  and S14 panels pixel-identical; 20 of 21 fast replots identical (S12b `theta_0` has the new band); the value of the
  regime in f is read from `results/fast/theta_0/t_3.0.pickle`, identical to the fast default run in every trial
  outcome, expert trial and final `w_ap`. Page version 2 (same URL as version 1, its source in
  `sandbox/results/scratch_2026-10-24_s15/page/`): https://claude.ai/artifact/17KgLcqMJ6zCfnTHY1VXaZ.
- S15 version 3 (user, 2026-10-24: "drop panel c. shift d to c and e to d. f becomes e and should be exactly two panels
  wide. in the theta_0 w_ap panel, please add a vertical line marking the chosen theta. in
  fig_s15e_theta_0_expert_trials you can drop the biological data. in this panel and the last one please make the picked
  parameter value an x instead of a larger dot"): the scatter of version 2's c (`fs15c_speedup`) is gone, d to f are c
  to e (`fig_s15c_theta_0_w_ap`, `fig_s15d_theta_0_expert_trials`, `fig_s15e_learning_rates`), so the figure is a, b, c
  in the first row and d with e (exactly two panels wide, 10.67 by 4.5 cm, as in version 2) in the second. c has a
  dashed line at `theta_0` 3 from `w_ap` 0 to 1 (black, `AXIS_WIDTH`, as f4g's expert-trial line; its axis now reaches 1
  plus the margin, as the biological S12b's does); S12b got the line the same day (user: "yes"; `plot_theta0_w_ap()`
  draws it in both regimes, 2.35 in the biological panel and 3 in the fast replot, which is now pixel-identical to S15c;
  apart from the line the biological S12b moves below a pixel: its y-limits are now exactly -0.05 to 1.05, where the
  minimum of the curves, 2e-6, had made them -0.049997). d shows the fast sweep only, on e's y-axis (0 to 1800 trials),
  with "Diverges" as in c and e. In d and e the regime's value is a cross (6 pt, line 1 pt, drawn above the markers next
  to it with a 0.4 pt white outline: at e's quarter-octave steps the markers overlap, and without the outline the cross
  disappeared among them). The axes of d and e have the same bottom and top (29.3 and 108.2 pt above the panel's bottom
  edge): e's label is an axis label centered under its four axes (`supxlabel` sat at 1 percent of the height and put e's
  axes 2.4 mm lower than d's), and the layout stand-ins (see Architecture) take up the rest (d's mathtext `theta_0`
  label is 2.5 pt deeper than plain text; e's titles take 7 pt above the axes); the stand-ins leave no text in the PDF
  or SVG (checked). Verified: a and b pixel-identical to version 2, c unchanged by the S12b line; e (before the
  alignment) differed from version 2's f only in four 37 by 36 px boxes around the crosses at 300 dpi; `python main.py
  plot` runs (6 min, again after the S12b line). Version 2's c to f are in
  `sandbox/results/scratch_2026-10-24_s15/v3/old_v2_files/`, its code in `v3/before/`. Page version 4 (same URL; the
  assembled figure at its true size, the five panels with legend drafts, the S12b replots with the line):
  https://claude.ai/artifact/17KgLcqMJ6zCfnTHY1VXaZ. User after page version 4 (2026-10-24): "everything looks pretty
  good for now". Decided (user, 2026-10-24): the TD-rate finding stays out of the text. Open for the user: the axis
  titles of e ("TD", "Policy", "Apical weights", "Apical input"; the manuscript's symbols if the methods define them;
  user: "TBD later"); still from version 1, the f4g / S13 window rule and keeping the fast `BU_*` rules out.

### Model state (decided in the sandbox)

- Environment: 1000 neurons, 30 distractors, basal weights in [0, 1], lognormal init (sigma 2, per-neuron
  normalization, total input 1.89 per neuron, the value at which one third of the neurons are texture-responsive by
  the manuscript's 0.1 criterion, clipped at 1), `lr_trace` 0.256 (irrelevant between 0.128 and 1.0).
- Biological regime = control4: `theta_0` 2.35, `lr_td` 0.00044, `lr_policy` 0.004, `lr_ap` 0.112. Smith 1129 / 1166
  on seeds 10-39 (no seed lost, chance during inhibition). 7 percent of non-selective and 3 percent of unresponsive
  neurons become selective (selective 13 to 22 percent), 25 percent at full gain. The ~50-trial gap after the lift on
  seeds 0-9 comes from seeds 1, 4, 5 and is accepted. Lessons: the TD rate sets the speed, `theta_0` the recruitment
  and the full-gain count; TD 0.0005 or less at policy 0.016 loses 5 to 10 percent of seeds to an always-lick
  lock-in, policy 0.004 removes it.
- Fast regime = control3: `theta_0` 3, `lr_td` 0.0035, `lr_policy` 0.016, `lr_ap` 0.512, the fastest control at this
  init (TD 0.004 or more, or `theta_0` 2, diverges; no margin).
- Rules (none gated by the apical activity, though calcium's level also reads it through `ca_beta`; each prunes, with
  an update set by the neuron's own gain-modulated drive against per-neuron thresholds, so it also runs at gain 1
  during apical inhibition): hebb4 (soft, `lr_bas` 0.512,
  `theta_bas_0` 7; 6 diverges), bcm4 (soft, `lr_bas` 0.128, `bcm_tau` 200, `bcm_e0_scale` 0.25; 0.3 or `bcm_tau` 300
  diverge), calcium4 (`pre`, `lr_bas` 0.016, `ca_ltd_frac` 0.4, `ca_theta_p` 1.5, `ca_gamma_d` 2.0, `ca_beta` 2). On
  seeds 10-39: hebb4 406 / 341, bcm4 475 / 496, calcium4 432 / 333. hebb3 (soft, 0.512, `theta_bas_0` 10), bcm3 (soft,
  0.256, `bcm_tau` 50, `bcm_e0_scale` 0.25) and calcium3 (`l1`, 0.001, `ca_ltd_frac` 0.5, `ca_theta_p` 2.0,
  `ca_gamma_d` 2.0, `ca_beta` 2); the slow-regime rule settings diverge at the fast control. With `ca_gamma_d` 2
  (calcium4, calcium3; 1.0 in calcium2) the net calcium update of an active synapse is never positive: -1.66 in the
  LTD window, -1.0 above `theta_p` (the potentiation term only offsets part of the depression), so relative growth
  comes only from the `pre` / `l1` renormalization. Set aside in September: `gated_hebb`, `burst`, `us` (explicit apical
  factor; they only re-weight the apically selected neurons and diverge via tone detectors at any rate that moves
  weight; re-tuned on control4 on 2026-10-18, where `gated_hebb` with `l1` and the un-gained drive is 41 to 64 percent
  faster than the control, see Figure review) and `oja` (no usable regime).
- Reading: after the lift hebb4 and calcium4 are faster than without inhibition (330 against 406 and 438 trials,
  rank-sum p 0.0025 and 0.028; sharpening during the 1800 blind trials), bcm4 is not (494 against 458, p 0.10), nor is
  the control; in the fast regime all three rules are (p 0.002, 0.002, 0.02). hebb and bcm end as sparse pruners (83 to
  86 percent unresponsive at control4); calcium keeps the control's recruitment pattern (61 percent unresponsive, 24
  percent selective); bcm at `bcm_tau` 50, `bcm_e0_scale` 0.5, `lr_bas` 0.128 (475 / 542) is the only bcm setting that
  keeps the recruitment.
- Reference sets the decision was made against: control2 (`theta_0` 4, `lr_td` 0.0008, `lr_policy` 0.016, `lr_ap`
  0.128; 952 / 1002, 1 percent recruited; its rules hebb2 394 / 232, bcm2 346 / 337, calcium2 564 / 357) and the
  2026-10-02 candidate (`theta_0` 2.25, `lr_td` 0.0005, `lr_policy` 0.004, `lr_ap` 0.128; 964 / 1066, 8.6 percent
  recruited, no figure set). Settings in `sandbox/results/s1_control4/finalists.json` and
  `sandbox/results/figures/<name>/hp.json`.

### Where things are

- Figure sets `sandbox/results/figures/<name>/` (`hp.json`, panels as PDF and PNG in git; `default.pickle` and
  `perturbed.pickle` local only, about 2 GB per set, 26 GB for the 13 sets `control2` to `calcium4` and `apical4`): the
  sets `*3`, `*4` and `apical4` are now duplicated bit for bit by `results/` and `results/fast/`; `*2` exist only there.
  Rebuild a set with `python -m sandbox.figures build <name> "$(cat .../hp.json)"` (then `plot <name>`, or `both`; about
  5 min per set). Comparison figures `compare_*` and `fs12cd_wide_*` (`python -m sandbox.selectivity <set names>`) and
  `speedup_slow_fast.png` (`python -m sandbox.speedup`) in `sandbox/results/figures/`; they are the prototypes of the
  new supplementary figures. `sandbox/results/` is gitignored; the files there that are in git were force-added.
- Sweep summaries (`summary.csv`, `grid.json`): `c10_*` to `c13_*`, `r5_*`, `o1_*` to `o3_*`, `fc1_*` to `fc4_*`;
  `python -m sandbox.tab <name>` prints them. Condition-4 scans (JSON lines per seed): `sandbox/results/s1_control4/`.
- Scratch scripts, configs, logs and page sources of the 2026-10-16 to 22 session, and the per-seed data of the apical
  rule's fast-regime scan (`fast/`): `sandbox/results/scratch_2026-10-16_22/`; the render and measurement scripts and
  the page sources of the 2026-10-22 to 23 session (S14, f4h_w_ap): `sandbox/results/scratch_2026-10-22_23/` (both
  described at the end of `sandbox/NOTES.md`); the fast-regime page (`page/fast_regime.html` with `img/`), its render
  script (`render.py`: PNGs through a patched `save_or_show`, `real` also writes the PDFs), number scripts and the code
  before the change (`before/`): `sandbox/results/scratch_2026-10-23_fast/`; S15 version 2 (2026-10-24): the page
  source (`page/`, the version-1 page as `fast_regime_v1.html`), the sweep and number scripts (`sim_lr.py`,
  `numbers_f.py` with `numbers_f.json`, `numbers_d.py`, `proto_f.py`), the code before the change (`before/`, with the
  dropped S15 panels) and the first draft's S15 PDFs (`old_s15_files/`): `sandbox/results/scratch_2026-10-24_s15/`,
  with version 3 in `v3/` (the code before it, version 2's PDFs and SVGs of c to f, the cross prototypes, the
  measurement and assembly scripts, `edit_panels.py`) and the S12b line in `v3/s12b_line/`.

### Next

1. Finalize the figures (user, 2026-10-16): continue the panel review with the user (f4g, f4e, f4f, S11e and f4h_w_ap
   done; still to review f4c, f4d, the f4h performance and expert-trial panels, S11b to S11d, S12 and S13; see Figure
   review). S14, the bottom-up plasticity figure of the biological regime, shows the apical rule alone (`BU_APICAL`,
   ported 2026-10-22, see the final-rule bullet) and was approved by the user on 2026-10-23 (b to f; a, the schematic of
   the rule, is still to be drawn; see Figure review). hebb, bcm and calcium have no panels since then (their
   `plot_bu_*` helpers, the five-condition PDFs in `panels/s14_rules_draft/` and their results are kept), so their open
   decisions (`ca_beta`, see the equations bullet; the re-tuned gated rules of 2026-10-18; the framing after the
   2026-10-19 ablation, the rules' speed-up does not need the apical gain) matter only for a fast-regime rules figure or
   if they return. The fast regime (2026-10-23): the control panels are replotted in `panels/fast/`, and S15 shows the
   fast regime in the user's layout of 2026-10-24 (version 3: a to e, e the new learning-rate sweeps; see the
   fast-regime bullets of Figure review); the user finds version 3 "pretty good for now" (2026-10-24, page version 4;
   open: the axis titles of e, "TBD later"). A fast-regime rules figure (hebb3, bcm3, calcium3) is not drawn and not
   recommended: it would come from the `plot_bu_*` helpers with `fast=True` (its panel c needs `x_max` 500 and e a
   smaller range; calcium3 is not faster than the control without inhibition, 213 against 230, p 0.29, and ends the
   inhibition protocol with fewer texture detectors than it started with, 4.5 to 3.3 percent, the loss coming after the
   lift), and the apical rule has no fast-regime setting (2026-10-22; a slower fast control, TD 0.002, would mean
   re-running hebb3, bcm3 and calcium3).
2. Normalization (user, 2026-10-14): investigate how normalization shapes the bottom-up plasticity. Current use: the
   init normalizes each neuron's lognormal weights to a total of 1.89 and clips at 1 (17 percent of the neurons
   clipped, mean total 1.84); during plasticity only calcium renormalizes (`pre` in calcium4, `l1` in calcium3, at
   every step), and since its net update is never positive the renormalization is its only source of growth; hebb and
   bcm use the soft bound and shrink the total input (hebb4 1.84 to 1.50, bcm4 to 1.17; fast 0.77 and 1.04). Plan:
   each rule with no normalization (soft or clamp), `pre`, `l1` and both, in both protocols, Smith-scored (expert
   trials, divergence, weight totals, selectivity), with a short `lr_bas` check per bound. Under `pre` the number of
   detectors of each stimulus equals its conserved total weight (about 56 neurons per texture), see the `fs14e`
   mechanism in Figure review.
3. Results hosting (user, 2026-10-24): the results stay out of git (8.3 GB against GitHub's free 10 GiB of LFS storage
   and 10 GiB of monthly bandwidth); once the figures are final, archive them (Zenodo proposed) and link the archive in
   the README, which for now says to run `python main.py simulate`. Upstream's `clean` branch (the fork has none): ask
   where it lives, then update it. Slimming (user, 2026-10-16: only once the figures are final): never read are
   `gain` outside the biological `DEFAULT` (1.8 GB), `w_ap`
   outside it beyond neurons 0 to 2 and the last trial (1.8 GB; the biological `APICAL_INHIBITION` needs neurons 1 and
   2), `x_som_texture` in the 8 inhibition runs (1.3 GB) and `w_bas_final` (a copy of `w_bas` or of the last snapshot,
   21 MB): 7.0 to 2.1 GB for the 16 runs before `BU_APICAL`, with `DEFAULT` (234 MB) and the rule files (139 to 145 MB)
   still above GitHub's 100 MB. f4g reads `x_pre_t` and `w_ap` of `DEFAULT`; `td_delta` is read by nothing, and neither
   is `sensory_dendrite` (the go/no-go mean, now only the reference of `get_sensory_dendrites()`), and `x_som_cue` is
   only read in `DEFAULT`. Changed since that list (2026-10-23): f4h_w_ap reads all of `w_ap` and the first 100 trials
   of `x_som_texture` of `DEFAULT` and `APICAL_INHIBITION`; S14 reads of both apical runs `performance` and `expert_t`,
   and of the unperturbed one also `outcome`, `x_som_texture`, `w_bas`, `w_bas_history` and the last row of `w_ap`; no
   panel reads the hebb, bcm and calcium runs. Since 2026-10-23 `panels/fast/` reads the fast `DEFAULT` and
   `APICAL_INHIBITION` as the biological panels read theirs; S15 (version 3, 2026-10-24) reads `performance` and
   `expert_t` of both regimes' `DEFAULT` and `APICAL_INHIBITION`, the fast `theta_0` sweep and the learning-rate sweeps
   (no longer the biological sweep, which S12b reads, nor `outcome_dendrite` or `v_hat`). The LFS pointers of the
   pre-port `results_default.pickle` and `results_perturbed.pickle` were removed from git on 2026-10-29, those of the
   deleted `MIXED_SELECTIVITY` and `BU_PLASTICITY` runs on 2026-10-16.
4. Decide what happens to the 26 GB of figure-set pickles (user, after the port), and delete the sandbox once the new
   figures exist (its prototypes read those pickles). The 9 sets `*3`, `*4` and `apical4` (18 GB) duplicate `results/`
   and `results/fast/`, the 4 sets `*2` (8 GB) exist only there; the duplicates can go once the new figures read
   `results/` (proposed 2026-10-16, awaiting the user; the disk is 70 percent full, 29 GB free on 2026-10-22).
5. Proposed cleanups awaiting the user (2026-10-16): move the port history above (verification, expert-trial table,
   `theta_0` scan) to `sandbox/NOTES.md` and keep a short summary here; remove `time_to_time_step()` and
   `time_step_to_time()` of `helper.py` (unused, added by the user in 46fb15e); delete `panels/f4g_options/` and
   `panels/theta0_options/` (gitignored option renders; the decision pages keep their own copies).
6. Manuscript text (outside the code): the default model is now the mixed one (1000 neurons, lognormal init, separate
   learning rates, `theta_0` 2.35), S12 shows it (fs12a equals f4c), the `theta_0` points 4 to 5 of S12b are not
   converged, and the calcium settings never give a net positive update of an active synapse.
7. Commit only when the user says so. Committed and pushed on 2026-10-24 (user: "commit and push"): the S14 draft,
   the f4h_w_ap groups, the `BU_APICAL` port, the fast regime with S15 version 3, the S12b `theta_0` line and the
   learning-rate sweeps, the sandbox knobs of the gated and apical rules (with `sandbox/scan_gated.py` and
   `sandbox/tab_gated.py`) and these notes. Committed and pushed on 2026-10-29 (user): the results out of git (see
   Data files: `/results/` in `.gitignore`, the README's Execution section, the two pre-port LFS pointers removed).
