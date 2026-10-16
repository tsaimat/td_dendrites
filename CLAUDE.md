# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Simulation code and figure generation for the manuscript "Error representations in apical dendrites of neocortical
layer 5 neurons during learning" (Schoenfeld et al., bioRxiv 10.1101/2021.12.28.474360). A mouse agent learns a go/no-go
texture discrimination task; 1000 L5 pyramidal neurons receive mixed-selective basal input (lognormal weights from the
three task stimuli and 30 distractors) and a top-down TD-error signal onto their apical dendrites, which
multiplicatively gates somatic output. Variants add bottom-up (basal) plasticity (`hebb`, `bcm`, `calcium`) and a
fast-learning regime. There is no test suite; correctness is checked by regenerating the panels in `panels/` and
comparing them to the manuscript figures, and the trial loop against the sandbox mirror
(`python -m sandbox.check_reproduces`).

## Environment and commands

Python 3.10, dependencies pinned in `requirements.txt` (torch 1.13, numpy 1.23, scipy 1.9, matplotlib 3.7,
statsmodels). Build the env as the README says (`conda create -n tdd python=3.10.8`, then `pip install -r
requirements.txt`) and use only that `tdd` env. No Matlab: `helper.smith_analysis()` ports the Smith et al. (2004)
estimator of the performance traces.

```
python main.py plot          # (default) plot all panels to panels/ and panels/svg/ from results/*.pickle
python main.py simulate      # rerun every simulation, including the performance traces (34 min on 8 cores)
python main.py perf          # extract the performance traces of the saved results again
```

`simulate` runs the 16 runs of `get_runs()`, then the `theta_0` sweep. Each `simulate_seeds()` call runs its 10 seeds in
parallel processes with one torch thread each (results do not depend on the thread count), adds each seed's Smith
performance trace and learning and expert trials (about 20 s for a 1800-trial seed, 50 s for a 4000-trial one, more
than the simulation itself) and writes its pickle once, complete, so an interruption leaves the files written before
it intact.

To plot a single panel interactively, call its function directly, e.g. `python -c "from l5apical.panels import *;
f4h_performance(saving=False)"`. Every panel function takes `saving: bool`; `False` shows the figure instead of
writing it. `python -m l5apical.panels` runs the ad-hoc block at the bottom of `panels.py` (currently plots everything
with `saving=False`).

To run one simulation: `l5apical.simulations.simulate_seeds(Simulation.X, fast=..., inhibition=...)` (`inhibition`
only for the `BU_*` members, `Simulation.APICAL_INHIBITION` is the inhibited control; `theta_0=...` gives a sweep point
of `DEFAULT`).

## Data files

- `results/` holds one pickle per run, each a list of `N_SEEDS` (10) dicts keyed by the `K_*` constants of
  `helper.py` (`KEYS_1D` per-trial vectors, `KEYS_2D` trials x (time steps or neurons)). Biological regime:
  `results_default.pickle` (`DEFAULT`), `results_perturbed.pickle` (`APICAL_INHIBITION`),
  `results_{hebb,bcm,calcium}.pickle` and `results_{hebb,bcm,calcium}_perturbed.pickle` (`BU_*` without and with
  apical inhibition). `results/fast/` holds the same eight files for the fast-learning regime (read only by its own
  supplementary figure), `results/theta_0/t_<theta_0>.pickle` the `theta_0` sweep of `DEFAULT`. `get_path()` is the
  single source of truth for these paths.
- Slim format (user, 2026-10-13): the trials x neurons arrays (`w_ap`, `gain`, `x_som_texture`) are float32, the
  eligibility trace `z` is not stored, and the sweep files keep only `outcome`, `correct_trials`, `w_bas`, the last
  row of `w_ap` and the performance. `performance` is the Smith trace (one entry more than the trials), `learning_t`
  and `expert_t` are ints counted from 0 (nan if not reached). `w_bas` holds the initial basal weights (the S12 panels
  bin neurons by them), `w_bas_final` the final ones, and the `BU_*` runs add `w_bas_history` (snapshots after trials
  0, 50, ..., last) and `w_bas_history_t`. `gain` is recorded at the time steps when each neuron's preferred stimulus
  (largest initial weight) is present, in every protocol. Sizes: 235 MB (`DEFAULT`), 518 MB (`APICAL_INHIBITION`), 284
  / 625 MB (`BU_*` without / with inhibition), 7.0 GB for the 16 runs, 1.8 MB per sweep value.
- `*.pickle` are git LFS objects (`.gitattributes`). git-lfs is not installed on this machine: the checkout held
  134-byte pointers, and the regenerated results are local and uncommitted. Never `git add` them without git-lfs
  (GitHub rejects files over 100 MB, and the history would carry them).
- Everything Matlab was removed on 2026-10-14 (user): the `Smith/` folder with the original Smith et al. scripts,
  `getperfs.m` and the `.mat` files of the trial outcomes and performances. Git history keeps them up to 390281e.
- The `clean` branch (the same code without results) is not on the fork: origin only has `main`.

## Architecture

`main.py` is a thin dispatcher into the `l5apical` package:

- `helper.py`: model constants (neuron and distractor counts, the lognormal init `INIT_SIGMA`, `INIT_SCALE`,
  `INIT_CLIP`, trial timing in time steps at `HZ` = 4 steps/s), `HYPERPARAMETERS` (`theta_0` and the learning rates of
  the policy, TD estimator, apical weights and apical afferent trace for the `biological` and `fast` regimes),
  `RULE_HYPERPARAMETERS` (bound, `lr_bas` and rule constants per regime and `BU_*` member), the `Simulation` and
  `Outcome` enums, the apical transfer and gain functions, `get_path()` (maps a simulation and its `fast`,
  `inhibition` and `theta_0` options to its results file), `get_runs()` (the 16 runs), `get_run_name()`, and
  `smith_analysis()`, the Python port of the learning-curve estimator of Smith et al. (2004) (`runanalysisv3.m` and its
  helpers; exact logit-normal quantiles instead of the Monte Carlo ones; validated against the Matlab output: identical
  learning and expert trials, traces within 3e-3).
- `simulations.py`: `Agent` is the policy network that picks lick/no-lick from somatic activity and does a
  REINFORCE-style update on reward. `get_params()` draws the apical weights, the distractor probabilities and the
  lognormal basal weights (L1-normalized per neuron to `INIT_SCALE`, clipped at 1), sets the apical plasticity
  thresholds (`theta_0` times the expected basal drive per time step, plus `BKG`) and relabels the neurons so that 0,
  1, 2 are the best tone, T2 and T1 neurons (the panels show them as cue, no-go and go neuron). `simulate_seed()` is
  the per-seed trial loop: basal activation, apical gain from `w_ap * x_pre_t`, somatic output, TD value estimate and
  apical plasticity gated by the per-neuron thresholds; apical inhibition silences the outcome-time apical activity
  for the first `N_TRIALS` of `N_TRIALS_AP_INH` trials. The `BU_*` members apply `basal_rule()` and
  `bound_basal_weights()` at every stimulus time step and once at the outcome (without input: the rule's thresholds
  see the outcome activity and the `pre` / `l1` bounds renormalize). `simulate_seeds()` runs the seeds in parallel
  processes (`simulate_and_analyse_seed()`), adds their performance (`get_performance()`) and pickles the list;
  `extract_performances()` redoes the performance of all saved results; `run()` does everything.
- `panels.py`: `get_results()` loads a pickle and optionally aligns all traces to each seed's expert trial
  (`align_to_expert()`: the `N_BEFORE_EXPERT` (500) trials before it and the `N_EXPERTS` (150) from it on, nan outside
  the simulation). `get_texture_responsive()` is the responsiveness test of S12d (mean somatic activity in the texture
  window above 0.1 over a range of trials), `get_sensory_dendrites()` computes the sensory-dendrite activity of any
  group of neurons from the stored `w_ap` and `x_pre_t` (exact to 1e-7 against the recorded go/no-go trace). Shared
  helpers (`plot_trace`, `plot_v_hat`, `plot_apical_raster`, `plot_apical_trace_boxplot`, `set_style`,
  `save_or_show`) are wrapped by thin per-panel functions named after manuscript figure panels (`f4c_...`,
  `fs12b_...`). `plot_all()` is the ordered list of every panel. All panels read the biological `DEFAULT` and
  `APICAL_INHIBITION` runs (S12 included, so `fs12a` plots the same as `f4c`), `fs12b_theta0` the sweep; nothing plots
  the `BU_*` or fast runs yet. Figure dimensions are in cm-derived constants at the top of the file and output is PDF
  at `DPI` plus an SVG copy.
- `parula.py`: Matlab parula colormap data.

Adding a new simulation variant means: an enum member in `Simulation` (and in `BU_SIMULATIONS` for a basal rule), its
name in `get_rule_name()` / `get_run_name()`, its constants in `RULE_HYPERPARAMETERS`, a branch in `basal_rule()` and
an entry in `get_runs()`.

## Work in progress (as of 2026-10-16)

Goal: a manuscript figure showing that apical top-down input can support bottom-up (basal) plasticity that speeds
learning beyond the apical loop alone. The exploration was done in `sandbox/` (see `sandbox/README.md`); the
chronological notes of everything tried, including the rules that were set aside, are in `sandbox/NOTES.md`.

### Port into the main code (done 2026-10-13 and 14)

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
- Still to review: the other panels of Fig. 4 and S11 to S13. The f4e rasters and f4f show single neurons (the cue,
  go and no-go neuron, 0, 2 and 1); the user's principle for f4g was that the model neurons should match a random
  sample of L5 neurons.

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
- Rules (no explicit apical factor; each prunes, with an update set by the neuron's own gain-modulated drive against
  per-neuron thresholds, so it also runs at gain 1 during apical inhibition): hebb4 (soft, `lr_bas` 0.512,
  `theta_bas_0` 7; 6 diverges), bcm4 (soft, `lr_bas` 0.128, `bcm_tau` 200, `bcm_e0_scale` 0.25; 0.3 or `bcm_tau` 300
  diverge), calcium4 (`pre`, `lr_bas` 0.016, `ca_ltd_frac` 0.4, `ca_theta_p` 1.5, `ca_gamma_d` 2.0, `ca_beta` 2). On
  seeds 10-39: hebb4 406 / 341, bcm4 475 / 496, calcium4 432 / 333. hebb3 (soft, 0.512, `theta_bas_0` 10), bcm3 (soft,
  0.256, `bcm_tau` 50, `bcm_e0_scale` 0.25) and calcium3 (`l1`, 0.001, `ca_ltd_frac` 0.5, `ca_theta_p` 2.0,
  `ca_gamma_d` 2.0, `ca_beta` 2); the slow-regime rule settings diverge at the fast control. With `ca_gamma_d` 2
  (calcium4, calcium3; 1.0 in calcium2) the net calcium update of an active synapse is never positive: -1.66 in the
  LTD window, -1.0 above `theta_p` (the potentiation term only offsets part of the depression), so relative growth
  comes only from the `pre` / `l1` renormalization. Set aside: `gated_hebb`, `burst`, `us` (explicit apical factor;
  they only re-weight the apically selected neurons and diverge via tone detectors at any rate that moves weight) and
  `oja` (no usable regime).
- Reading: every rule is faster after the lift than without inhibition (sharpening during the 1800 blind trials), the
  control is not. hebb and bcm end as sparse pruners (83 to 86 percent unresponsive at control4); calcium keeps the
  control's recruitment pattern (61 percent unresponsive, 24 percent selective); bcm at `bcm_tau` 50, `bcm_e0_scale`
  0.5, `lr_bas` 0.128 (475 / 542) is the only bcm setting that keeps the recruitment.
- Reference sets the decision was made against: control2 (`theta_0` 4, `lr_td` 0.0008, `lr_policy` 0.016, `lr_ap`
  0.128; 952 / 1002, 1 percent recruited; its rules hebb2 394 / 232, bcm2 346 / 337, calcium2 564 / 357) and the
  2026-10-02 candidate (`theta_0` 2.25, `lr_td` 0.0005, `lr_policy` 0.004, `lr_ap` 0.128; 964 / 1066, 8.6 percent
  recruited, no figure set). Settings in `sandbox/results/s1_control4/finalists.json` and
  `sandbox/results/figures/<name>/hp.json`.

### Where things are

- Figure sets `sandbox/results/figures/<name>/` (`hp.json`, panels as PDF and PNG in git; `default.pickle` and
  `perturbed.pickle` local only, about 2 GB per set, 24 GB for the 12 sets `control2` to `calcium4`): the sets `*3`
  and `*4` are now duplicated bit for bit by `results/` and `results/fast/`; `*2` exist only there. Rebuild a set with
  `python -m sandbox.figures build <name> "$(cat .../hp.json)"` (then `plot <name>`, or `both`; about 5 min per set).
  Comparison figures `compare_*` and `fs12cd_wide_*` (`python -m sandbox.selectivity <set names>`) and
  `speedup_slow_fast.png` (`python -m sandbox.speedup`) in `sandbox/results/figures/`; they are the prototypes of the
  new supplementary figures. `sandbox/results/` is gitignored; the files there that are in git were force-added.
- Sweep summaries (`summary.csv`, `grid.json`): `c10_*` to `c13_*`, `r5_*`, `o1_*` to `o3_*`, `fc1_*` to `fc4_*`;
  `python -m sandbox.tab <name>` prints them. Condition-4 scans (JSON lines per seed): `sandbox/results/s1_control4/`.

### Next

1. Finalize the figures (user, 2026-10-16): continue the panel review with the user (f4g done, see Figure review),
   then design the two new supplementary figures (the rules in the biological regime, and the fast regime): performance
   traces and expert trials in both protocols, selectivity before and after, detector counts, from
   `results/results_{hebb,bcm,calcium}*.pickle` and `results/fast/`; prototypes `sandbox/figures.py`,
   `selectivity.py` and `speedup.py`.
2. Normalization (user, 2026-10-14): investigate how normalization shapes the bottom-up plasticity. Current use: the
   init normalizes each neuron's lognormal weights to a total of 1.89 and clips at 1 (17 percent of the neurons
   clipped, mean total 1.84); during plasticity only calcium renormalizes (`pre` in calcium4, `l1` in calcium3, at
   every step), and since its net update is never positive the renormalization is its only source of growth; hebb and
   bcm use the soft bound and shrink the total input (hebb4 1.84 to 1.50, bcm4 to 1.17; fast 0.77 and 1.04). Plan:
   each rule with no normalization (soft or clamp), `pre`, `l1` and both, in both protocols, Smith-scored (expert
   trials, divergence, weight totals, selectivity), with a short `lr_bas` check per bound.
3. Results hosting: committing the 7 GB needs git-lfs here and LFS quota (the README already says `main` carries about
   7 GB in LFS), or the results go elsewhere. The `clean` branch: ask where it lives, then update it. Slimming (user,
   2026-10-16: only once the figures are final): never read are `gain` outside the biological `DEFAULT` (1.8 GB),
   `w_ap` outside it beyond neurons 0 to 2 and the last trial (1.8 GB; the biological `APICAL_INHIBITION` needs neurons
   1 and 2), `x_som_texture` in the 8 inhibition runs (1.3 GB) and `w_bas_final` (a copy of `w_bas` or of the last
   snapshot, 21 MB): 7.0 to 2.1 GB, with `DEFAULT` (234 MB) and the rule files (139 to 145 MB) still above GitHub's
   100 MB. f4g reads `x_pre_t` and `w_ap` of `DEFAULT`; `td_delta` is read by nothing, and neither is
   `sensory_dendrite` (the go/no-go mean, now only the reference of `get_sensory_dendrites()`). In git,
   `results_default.pickle` and `results_perturbed.pickle` are still the LFS pointers of the pre-port results; the
   pointers of the deleted `MIXED_SELECTIVITY` and `BU_PLASTICITY` runs were removed on 2026-10-16.
4. Decide what happens to the 24 GB of figure-set pickles (user, after the port), and delete the sandbox once the new
   figures exist (its prototypes read those pickles). The 8 sets `*3` and `*4` (16 GB) duplicate `results/` and
   `results/fast/`, the 4 sets `*2` (8 GB) exist only there; the duplicates can go once the new figures read
   `results/` (proposed 2026-10-16, awaiting the user; the disk is 62 percent full, 37 GB free).
5. Proposed cleanups awaiting the user (2026-10-16): move the port history above (verification, expert-trial table,
   `theta_0` scan) to `sandbox/NOTES.md` and keep a short summary here; remove `time_to_time_step()` and
   `time_step_to_time()` of `helper.py` (unused, added by the user in 46fb15e); delete `panels/f4g_options/` and
   `panels/theta0_options/` (gitignored option renders; the decision pages keep their own copies).
6. Manuscript text (outside the code): the default model is now the mixed one (1000 neurons, lognormal init, separate
   learning rates, `theta_0` 2.35), S12 shows it (fs12a equals f4c), the `theta_0` points 4 to 5 of S12b are not
   converged, and the calcium settings never give a net positive update of an active synapse.
