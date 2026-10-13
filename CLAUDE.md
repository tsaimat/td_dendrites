# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Simulation code and figure generation for the manuscript "Error representations in apical dendrites of neocortical layer 5 neurons during learning" (Schoenfeld et al., bioRxiv 10.1101/2021.12.28.474360). A mouse agent learns a go/no-go texture discrimination task; L5 pyramidal neurons receive basal sensory input (task stimuli plus many distractors) and a top-down TD-error signal onto their apical dendrites, which multiplicatively gates somatic output. There is no test suite; correctness is checked by regenerating the panels in `panels/` and comparing them to the manuscript figures.

## Environment and commands

Python 3.10, dependencies pinned in `requirements.txt` (torch 1.13, numpy 1.23, scipy 1.9, matplotlib 3.7, statsmodels). Build the env as the README says (`conda create -n tdd python=3.10.8`, then `pip install -r requirements.txt`) and use only that `tdd` env. Matlab (tested with R2022b) is only needed to regenerate performance traces.

```
python main.py plot       # (default) plot all panels to panels/ and panels/svg/ from results/*.pickle
python main.py simulate   # rerun every simulation, overwriting results/ (slow; CPU torch)
python main.py perf       # merge Smith/performances.mat back into the results pickles
```

Full regeneration pipeline, in order: `simulate` -> run `Smith/getperfs.m` in Matlab -> `perf` -> `plot`. `simulate` ends by calling `save_trial_outcomes_matlab()` (writes `Smith/outcomes_*.mat`) and, if `Smith/performances.mat` already exists, `load_smith_perf()`. Skipping or interrupting steps corrupts the pickles; recovery is rerunning from scratch or checking out the committed results.

To plot a single panel interactively, call its function directly, e.g. `python -c "from l5apical.panels import *; f4h_performance(saving=False)"`. Every panel function takes `saving: bool`; `False` shows the figure instead of writing it. `python -m l5apical.panels` runs the ad-hoc block at the bottom of `panels.py` (currently plots everything with `saving=False`).

To run one simulation type: `l5apical.simulations.simulate_seeds(Simulation.X)`; `play.py` is a scratch script for this.

## Data files

- `results/*.pickle` are git LFS objects (~1.5 GB total). Each is a list of `N_SEEDS` (10) dicts keyed by the `K_*` string constants in `helper.py`. `KEYS_1D` entries are per-trial vectors, `KEYS_2D` are trials x (time steps or neurons).
- `results/mixed_selectivity/t_{theta_0}.pickle` holds one file per theta_0 in `get_thetas_0()`.
- `Smith/` contains the Matlab state-space performance estimator from Smith et al. 2004. `getperfs.m` reads `outcomes_*.mat` and writes `performances.mat` containing `perf_*` traces plus `expert_t` and `learning_t`. The ordering of simulations in `getperfs.m` (`lfilenames`) must match the `simulations` list in `load_smith_perf()`, which indexes `expert_t`/`learning_t` by position.
- The `clean` branch is the same code without the committed results, for users who want to simulate from scratch.

## Architecture

`main.py` is a thin dispatcher into the `l5apical` package:

- `helper.py`: all model constants (learning rate, thresholds, trial timing in time steps at `HZ` = 4 steps/s, neuron counts), the `Simulation` and `Outcome` enums, the apical transfer/gain functions, and `get_path()`, the single source of truth mapping a `Simulation` to its results file. Both simulation and plotting resolve files through it, so redirecting a simulation's output (as the current working copy does for `DEFAULT` -> `results_bup.pickle`) changes what every panel reads.
- `simulations.py`: `Agent` is the policy network that picks lick/no-lick from somatic activity and does a REINFORCE-style update on reward. `simulate_seed()` is the per-seed trial loop: each trial steps from tone through texture to outcome, computes basal activation, apical gain from `w_ap * x_pre_t`, somatic output, a TD value estimate, and plasticity on the apical weights gated by the per-neuron `thetas`. Simulation types branch inside this one function: `APICAL_INHIBITION` runs `N_TRIALS_AP_INH` trials with apical input off for the first `N_TRIALS`; `MIXED_SELECTIVITY` uses a random non-diagonal `w_bas` and `theta_0`-scaled thresholds from `get_params()`; `BU_PLASTICITY` uses the same random L1-normalized `w_bas` init as the mixed model but with `N_Z_BU_PLASTICITY` distractors, and additionally lets `w_bas` learn with a soft-bounded `(1 - w) * w` rule, so weights stay in [0, 1] (an identity init would make that rule a no-op). `simulate_seeds()` loops seeds and pickles the list.
- `panels.py`: `get_results()` loads a pickle and optionally aligns all traces to each seed's expert trial. Shared helpers (`plot_trace`, `plot_v_hat`, `plot_apical_raster`, `plot_apical_trace_boxplot`, `set_style`, `save_or_show`) are wrapped by thin per-panel functions named after manuscript figure panels (`f4c_...`, `fs12b_...`). `plot_all()` is the ordered list of every panel. Figure dimensions are in cm-derived constants at the top of the file and output is PDF at `DPI` plus an SVG copy.
- `parula.py`: Matlab parula colormap data.

Adding a new simulation variant means: add an enum member in `Simulation`, a path in `get_path()`, parameter handling in `get_params()`, any per-trial branch in `simulate_seed()`, a call in `run()`, an entry in `save_trial_outcomes_matlab()` and `load_smith_perf()`, and a matching filename and `perf_*` variable in `Smith/getperfs.m`.

## Work in progress (as of 2026-10-13)

Goal: a manuscript figure showing that apical top-down input can support bottom-up (basal) plasticity that speeds
learning beyond the apical loop alone. The exploration was done in `sandbox/` (see `sandbox/README.md`); the
chronological notes of everything tried, including the rules that were set aside, are in `sandbox/NOTES.md`. The
main `l5apical` code is untouched since the `BU_PLASTICITY` scaffold was added: its rule (`hebb`, soft bound, no
basal threshold) and `results/results_bup.pickle`, `Smith/outcomes_bup.mat` and the `perf_bup` rows of
`Smith/performances.mat` are stale no-op-equivalent data and must be regenerated after the port.

Decisions (user, 2026-10-13): control4 (condition 4 below) is the final control. It becomes the basis of `DEFAULT`
and `APICAL_INHIBITION`, so it produces all model figures, Fig. 4 and S11 to S13. `MIXED_SELECTIVITY` is deleted,
because the new default replaces it. `BU_PLASTICITY` gives way to `BU_HEBB`, `BU_BCM` and `BU_CALCIUM`, with the
settings of `hebb4`, `bcm4` and `calcium4`, for a new supplementary figure. The port is the next step (plan below).
control4's ~50-trial gap after the lift on seeds 0-9 is accepted.

### Environment chosen (user decisions)

- 1000 neurons (`n_z`), 30 distractors (`n_noise`), basal weights hard-constrained to [0, 1].
- Lognormal initialization chosen on structural grounds: `init` lognormal, `init_sigma` 2, per-neuron normalized
  (`init_norm` neuron), total input 1.89 per neuron (`init_scale`, the value at which one third of the neurons are
  texture-responsive by the manuscript's 0.1 criterion), clipped at 1 (`init_clip`).
- Slow policy rate `lr_policy` 0.016 (faster policy rates learn slower or lose seeds at this init), `lr_trace` 0.256
  (irrelevant between 0.128 and 1.0).
- Two control conditions (no basal plasticity), both learning in every seed and at chance during apical inhibition:
  - Slow / robust: `theta_0` 4, `lr_td` 0.0008, `lr_ap` 0.128. Proxy 1038 trials without inhibition, 1090 after the
    lift; survives a 1.7-fold drive amplification (the margin covers the drive growth the rules produce).
  - Fast / at the edge: `theta_0` 3, `lr_td` 0.0035, `lr_ap` 0.512. Proxy 306 / 372; the fastest control possible
    at this init (TD 0.004 or more, or `theta_0` 2, diverges). No margin.
- Rules carried forward: `hebb` with its own basal threshold (`theta_bas_0`), `bcm`, `calcium`. Set aside:
  `gated_hebb`, `burst`, `us` (explicit apical factor; they only re-weight the apically selected neurons and diverge
  via tone detectors at any rate that moves weight) and `oja` (no usable regime). Every working rule is a pruner
  whose post-synaptic sign is set by the neuron's own gain-modulated drive against a per-neuron threshold, so the
  apical signal enters through `x_som` and the sharpening also runs at gain 1 (during apical inhibition).

### Finalists (10 seeds; Smith expert trials mean +- sd, no inhibition / after the lift; no divergence anywhere)

Slow condition (figure sets `*2`, all `lr_ap` 0.128):
- control2: 952 +- 263 / 1002 +- 246.
- hebb2: soft bound, `lr_bas` 0.256, `theta_bas_0` 8: 394 +- 65 / 232 +- 108.
- bcm2: soft, `lr_bas` 0.256, `bcm_tau` 200, `bcm_e0_scale` 0.5: 346 +- 69 / 337 +- 65.
- calcium2: `pre` bound, `lr_bas` 0.004, `ca_ltd_frac` 0.5, `ca_theta_p` 1.5, `ca_gamma_d` 1.0, `ca_beta` 2:
  564 +- 81 / 357 +- 90.

Fast condition (figure sets `*3`, all `lr_ap` 0.512; the slow finalists all diverge here, the rules need constants
that keep the population drive below the control's):
- control3: 230 +- 56 / 267 +- 85.
- hebb3: soft, `lr_bas` 0.512, `theta_bas_0` 10: 136 +- 23 / 85 +- 24.
- bcm3: soft, `lr_bas` 0.256, `bcm_tau` 50, `bcm_e0_scale` 0.25: 148 +- 26 / 104 +- 22.
- calcium3: `l1` bound, `lr_bas` 0.001, `ca_ltd_frac` 0.5, `ca_theta_p` 2.0, `ca_gamma_d` 2.0, `ca_beta` 2:
  213 +- 42 / 168 +- 38.

Reading: the pruning rules (`hebb`, `bcm`) cut the time to expert by about 60 percent at the slow control and 30 to
40 percent at the fast one, calcium by 40 and 7 percent; every rule is faster after the lift than without inhibition
(sharpening during the 1800 blind trials), the control is not. `hebb` and `bcm` end with about 90 percent
unresponsive neurons and 3 percent go and 3 percent no-go one-hot detectors at maximal gain (a sparse code);
`calcium` keeps a graded code (7 percent go, 7 percent no-go selective, 64 to 75 percent unresponsive); the control
keeps 42 percent texture-responsive with responses proportional to the initial basal selectivity.

### Condition 4 (2026-10-06): control with recruitment, tuned to 1100 trials

The slow control (`theta_0` 4) recruits almost no neurons: 1 percent of non-selective neurons become selective
(manuscript mixed model: 13 percent), because its thresholds sit above the texture response of weakly driven
neurons. The apparent loss of selective neurons in its fs12cd panel is a +-4 histogram-range artifact (gain 10
pushes response differences to 9); the panel needs a wider range or a pie over all neurons. Condition 4 lowers
`theta_0` and retunes the rates, scored with the real Smith estimator (`sandbox/smith.py` on the full series, as
the pipeline does: inhibition protocol over all 4000 trials, minus 1800). Settings in
`sandbox/results/s1_control4/finalists.json`, scan data (JSON lines per seed) in the same folder, scan scripts
`sandbox/scan_smith.py`, `scan_table.py`, `scan_rtable.py`.

- control4: `theta_0` 2.35, `lr_td` 0.00044, `lr_policy` 0.004, `lr_ap` 0.112, rest as above. Smith 1100 +- 259 /
  1154 +- 198 (seeds 0-9), 1129 / 1166 (seeds 10-39), no seed lost, chance during inhibition. 7 percent of
  non-selective and 3 percent of unresponsive neurons become selective (selective 13 to 22 percent), 25 percent at
  full gain. The ~50-trial gap after the lift on seeds 0-9 comes from seeds 1, 4, 5 and survived every combination.
- Lessons: TD rate sets the speed; `theta_0` sets recruitment and the full-gain count (2.0: 12 percent recruited,
  36 to 46 percent full gain; 2.5: 4.5 percent; 3: 2 percent); TD 0.0005 or less at policy 0.016 loses 5 to 10
  percent of seeds to an always-lick lock-in, policy 0.004 removes it; trace rate below 0.256 makes it worse.
- Rule finalists on control4 (Smith, seeds 0-9 / seeds 10-39, no inhibition / after the lift, no divergence):
  hebb4 (soft, `lr_bas` 0.512, `theta_bas_0` 7; 6 diverges) 406 / 330 and 406 / 341; bcm4 (soft, `lr_bas` 0.128,
  `bcm_tau` 200, `bcm_e0_scale` 0.25; 0.3 or `bcm_tau` 300 diverge) 458 / 494 and 475 / 496; calcium4 (`pre`,
  `lr_bas` 0.016, `ca_ltd_frac` 0.4, `ca_theta_p` 1.5, `ca_gamma_d` 2.0, `ca_beta` 2) 438 / 330 and 432 / 333.
  hebb and bcm again end as sparse pruners (83 to 86 percent unresponsive); calcium keeps the control's
  recruitment pattern (61 percent unresponsive, 24 percent selective). bcm at `bcm_tau` 50, `bcm_e0_scale` 0.5,
  `lr_bas` 0.128 (475 / 542, one late seed) is the only bcm setting that keeps the recruitment.
- Reference sets the decision was made against: control2 (above) and the 2026-10-02 candidate (`theta_0` 2.25, `lr_td`
  0.0005, `lr_policy` 0.004, `lr_ap` 0.128; Smith 964 / 1066 on seeds 0-9, 8.6 percent recruited).
- Figure sets `control4`, `hebb4`, `bcm4` and `calcium4` (built 2026-10-06, pushed as 3da2242 on 2026-10-13) and the
  comparisons `compare_control2_control4`, `fs12cd_wide_control2_control4`, `compare_control4_hebb4_bcm4_calcium4`,
  `fs12cd_wide_control4_hebb4_bcm4_calcium4`. The candidate has no figure set. Selectivity counts come from the
  `fs12cd_wide_*` figures; every set's own `fig_s12c_selectivity_distributions` panel still has the range artifact.
  User decision (2026-10-13): control4 for all main figures, the three rules for a new supplementary figure.

### Where things are

- Exact settings per figure set: `sandbox/results/figures/<name>/hp.json`; the pickles there (`default.pickle`,
  `perturbed.pickle`, about 2 GB per set, 24 GB for the 12 sets, local only) were built from the `hp.json` files
  (sets `*2` and `*3` rebuilt on 2026-10-01, sets `*4` on 2026-10-06) and reproduce the numbers above exactly
  (`python -m sandbox.figures build <name> "$(cat .../hp.json)"`, then `plot <name>` for the panels, or `both`; about
  5 min per set with 4 workers). Panels in `.../panels/` (PDF and PNG in git, SVG local). Comparison figures
  `compare_*` and `fs12cd_wide_*` (`python -m sandbox.selectivity <set names>`, from the pickles) and the
  two-condition summary `speedup_slow_fast.png` in `sandbox/results/figures/`. `sandbox/results/` is gitignored;
  the files there that are in git were force-added (`git add -f`).
- Sweep summaries kept (`summary.csv`, `grid.json`): `c10_*` to `c13_*` (control at 1000 neurons), `r5_*` (rules at
  the slow control), `o1_*` to `o3_*` (fastest control), `fc1_*` to `fc4_*` (rules at the fast control).
  `python -m sandbox.tab <name>` prints them. Everything from the 203-neuron passes was deleted (in git history).
- Sandbox knobs that are ablations only and need not be ported: `rule_sees_gain`, `reset_at_lift`, `lr_dep` /
  `dep_form`, `theta_gate`, `w_bas_scale`, `gh_*`, `burst_*`, `us_tau`, `oja_decay`. Figure helpers `sort_neurons`,
  `gain_at_preferred` and `store_full` are needed by the panels.

### Next: the port into the main code (design decided 2026-10-13)

1. `helper.py` and `simulations.py`: control4 becomes `DEFAULT` and `APICAL_INHIBITION`. That means 1000 neurons,
   30 distractors, the lognormal init (sigma 2, per-neuron normalization, scale 1.89, clip 1), separate rates
   (`lr_policy` 0.004, `lr_td` 0.00044, `lr_ap` 0.112, `lr_trace` 0.256 instead of the single `LR`) and `theta_0`
   2.35. Delete `MIXED_SELECTIVITY` (enum member, `get_path`, the `get_params` and `simulate_seed` branches, the
   `theta_0` loop in `run()`, `results/mixed_selectivity/`). Replace `BU_PLASTICITY` (and `N_Z_BU_PLASTICITY`) by
   `BU_HEBB`, `BU_BCM` and `BU_CALCIUM` with the rules, bounds and constants of `hebb4`, `bcm4` and `calcium4`
   (`sandbox/results/s1_control4/finalists.json`). The supplementary figure compares both protocols, so each rule
   also needs the apical-inhibition protocol. Whether that is separate members or a protocol argument is still open.
   Port the figure helpers the panels need (`store_full`, `sort_neurons`, `gain_at_preferred`).
2. Matlab bookkeeping: `save_trial_outcomes_matlab()`, `load_smith_perf()` and `Smith/getperfs.m` (`lfilenames`,
   `perf_*`) get the new list of simulations, in the same order (the mixed and bup entries go).
3. Panels: `fs12a_performance_eg`, `fs12b_wap_traces`, `fs12b_theta0`, `fs12c_selectivity_traces` and
   `fs12cd_selectivity_distribution` read `DEFAULT` instead of `MIXED_SELECTIVITY`. `fs12b_theta0` needs a
   `theta_0` sweep of the new default. The range of `get_thetas_0()` (1 to 8) probably needs adapting at this init:
   1.5 puts 91 percent of the neurons at full gain, and 4 recruits almost none.
4. Verify: the ported runs must match the sandbox mirror at the `finalists.json` settings, bit-identically if the
   random draws stay in the same order (repoint `sandbox/check_reproduces.py`, which compares at the old defaults).
   The Smith expert trials must match condition 4 above (control4 1100 / 1154 on seeds 0-9).
5. Regenerate every result: `python main.py simulate`, then `Smith/getperfs.m` or `sandbox/smith.py`, then
   `python main.py perf` and `plot`.
6. Design the new supplementary figure: performance traces and expert trials in both protocols, selectivity before
   and after, detector counts. The prototypes are `sandbox/figures.py`, `selectivity.py` and `speedup.py`.
7. Update this file's architecture and data sections, the README and the `clean` branch.
8. Later (user, 2026-10-13): fix `fs12cd_selectivity_distribution`. Its bins (`np.arange(-4, 4, 0.2)`, so -4 to
   3.8) drop every neuron outside that range from both the histogram and the pie. control4's responses differ by up
   to 9, so the new S12 panels need the fix: widen the range (`fs12cd_wide` in `sandbox/selectivity.py` uses +-10,
   the maximal gain) or count the pie over all neurons.
9. After the port (user, 2026-10-13): decide what happens to the 24 GB of local figure-set pickles.
