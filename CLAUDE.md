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

## Work in progress (as of 2026-09-29)

The `Simulation.BU_PLASTICITY` variant and the exploration sandbox are committed and pushed to the fork
(`tsaimat/td_dendrites`). State of the work:

- Main code is done: `N_Z_BU_PLASTICITY = 30` distractors, random L1-normalized `w_bas` init shared with the mixed model, soft-bounded basal plasticity `(1 - w) * w` in the trial loop, gain recording via a neuron-space mask, Matlab script preallocates from the file list. The default model reproduces the committed `results_default.pickle` exactly, and the mixed-model initialization is bit-identical to before.
- Data is stale: `results/results_bup.pickle`, `Smith/outcomes_bup.mat` and the `perf_bup` rows in `Smith/performances.mat` were generated when the variant was a no-op (identity init, so identical to default). Regenerate once a rule is chosen: `simulate_seeds(Simulation.BU_PLASTICITY)`, `save_trial_outcomes_matlab()`, `Smith/getperfs.m` in Matlab, then `python main.py perf`.
- No panel plots the bottom-up results yet. `f4f_sen_dendrite` assumes neuron i is wired to stimulus i, so it is only meaningful for the default model.
- `sandbox/` is a self-contained, deletable package for trying out bottom-up plasticity rules (see `sandbox/README.md`). Its `simulate.py` mirrors the main trial loop (including apical inhibition); run `python -m sandbox.check_reproduces` after editing either (passes). Sweep output goes to the gitignored `sandbox/results/`. Run everything in the `tdd` conda env (built per the README from `requirements.txt`); never use the machine's other envs.
- `play.py` is a scratch script and `Overview.pdf` is a manuscript overview; both are committed but were not part of the original repo.

### Sandbox findings (full-length sweeps, 5 seeds, Python expert-trial proxy)

- The rule in the main code (`hebb`, `x_in (x_som - theta)`, soft bound) diverges to NaN in every full-length run at `LR`, also with L1 normalization. Mechanism: the population drive per stimulus grows (soft bound: several synapses per neuron go to 1; L1: many neurons converge on the tone, the most frequent input), and the TD value estimator, tuned for one neuron per stimulus, blows up. This is intrinsic: even the no-plasticity model with a near one-hot random init (`init_power` 24, 14 distractors) diverges in 1 of 5 seeds. Any rule must keep the population drive per stimulus at a few units.
- With 30 distractors and the mixed-model init (`rand ** 6`, median strongest weight per neuron 0.19) the no-plasticity control never learns in 1800 trials. With `rand ** 12` (strongest weight 0.28) it learns in 1 of 5 seeds (proxy 1621 trials); the paper's 14-distractor mixed model learns in all seeds (1368). The bottom-up sweeps were run at `init_power = 12`.
- Reference (default model, identity init): expert proxy 1545 trials; under apical inhibition performance stays at 0.50 and expert comes 1549 trials after the lift.
- `calcium` (LTD/LTP thresholds) with L1 normalization at `lr_bas` 0.002 to 0.004 is the strongest rule: 5 of 5 seeds expert, 780 to 1100 trials, no divergence, robust to its threshold parameters as long as the LTD window is wide (`ca_ltd_frac` 0.3 to 0.5). Mechanism: a within-neuron winner-take-all that turns nearly every neuron into a one-hot detector of its dominant input (about 170 distractor and 10 to 20 texture detectors); the apical term of the calcium proxy (`ca_beta`) makes no difference. `bcm` with the soft bound at 0.064 does the same faster (637 trials) but diverges in 2 of 5 seeds.
- Because that sharpening is apical-blind, it also runs during apical inhibition: performance stays at chance (criterion 1 holds for every rule tested), but after the lift learning is faster than without inhibition (calcium 789 vs 1107 trials, bcm 477 vs 637), so criterion 2 (same time after the lift) is violated in that direction.
- The apically gated rules (`burst`, `gated_hebb`, `us`) are silent during inhibition and build texture detectors only once the apical signal exists, so they satisfy both criteria by construction, but they cannot bootstrap learning: at `lr_bas` 0.004 with the soft bound they turn 1 in 5 control seeds into 2 to 3 of 5 (1415 to 1607 trials), the detectors they form are mostly texture detectors (about 10 to 17 vs 8 to 15 distractor), and with a higher rate or the L1 bound the shared apical signal spreads potentiation over the whole population (task fraction rises, no neuron becomes a detector) and the value estimator diverges, in the 4000-trial protocol also late after learning.
- Constants grids (`hp_burst`, `hp_bcm`, `hp_gated_hebb`, `hp_us`, `hp_inh` in `sandbox/results`): `burst` is insensitive to its averaging time constant (1 to 100 trials all give 2 of 5 seeds at about 1400); its somatic-rate term diverges with the soft bound and only reaches 2 of 5 seeds with L1. `gated_hebb` is insensitive to the gate sharpness. `bcm` depends strongly on the sliding threshold: a fixed or slow (500 trials) threshold always diverges, a fast one (20 trials) with a lowered fixed point (`bcm_e0_scale` 2, lr 0.016) gives 5 of 5 seeds at 1401 trials without divergence, but the total drive per neuron grows 55 percent and under inhibition it sharpens like calcium (1069 after the lift, 1 divergence). `us` with a running-average gain as prediction (`us_tau` 5 trials, L1, lr 0.004) goes from 0 to 4 of 5 seeds but only at 1593 trials, and the detectors it forms are distractor ones. The best calcium setting (`ca_ltd_frac` 0.3, `ca_theta_p` 1.5, lr 0.004, apical term off) reaches 780 to 820 trials in 5 of 5 seeds, and 495 after the lift.
- Environment sweep (`env_control`, `env_rules`, `env_rules2`, `env_inh`; sandbox knobs `n_z`, `init_power`, `lr_ap`,
  `lr_policy`, main-code constants untouched): the no-plasticity control at `init_power` 6 never learns for any
  neuron count (100 to 800) or learning rate (0.016 to 0.064). At `init_power` 12 the control depends on the
  population size: 100 neurons 0 of 5, 203 neurons 1 of 5 (3 of 5 with both learning rates at 0.064), 400 and 800
  neurons 5 of 5 (1586 to 790 trials, faster with higher `lr_ap` and `lr_policy`, both help about equally; no
  divergence even at 800 neurons). `init_power` 9 is intermediate (203 neurons 1 of 5, 400 neurons 2 of 5 at the fast
  rates). Faster apical or policy learning never rescues a regime the control fails in.
- Plasticity rules vs population size: they pull in opposite directions. `calcium` needs a small population: 5 of 5
  at 203 neurons (`init_power` 12, 793 trials; `init_power` 9, 1039) and at 100 neurons (`init_power` 12, 894, where
  the control is 0 of 5), 2 to 3 of 5 at 50 neurons or at `init_power` 6, and it diverges in every seed at 400 or 800
  neurons because it builds 20 to 35 tone detectors. The gated rules (`gated_hebb`, `burst`, `us`) get worse with
  more neurons: at 800 neurons, where the control learns in 5 of 5, they diverge in every seed (about 30 texture
  detectors form from the shared apical signal), and at `init_power` 6 they stay silent at any size. Higher `lr_ap`
  and `lr_policy` make gated rules diverge more (203 neurons, `init_power` 12: 2 divergences vs 0), and give calcium
  nothing (793 vs 780).
- Inhibition protocol at the new settings: calcium at 100 neurons stays at chance and reaches expert 498 trials after
  the lift (894 without inhibition), so the criterion 2 asymmetry persists; the control at 800 neurons with fast
  rates is 1001 after the lift vs 790 without.
- Gated Hebb in small populations (`gh_small`, `gh_theta`, `gh_boot`; 20 to 100 neurons, `init_power` 6 and 12,
  both bounds, `lr_bas` 0.004 to 0.064, `theta_0` 0.25 to 4, `lr_td` 0.016 to 0.25, `lr_ap` 0.016 to 0.064): every
  setting stays at chance or diverges. Mechanism (one-seed diagnostics in `sandbox/results/_diag.py`): with a dense
  init the strongest texture weight is about 0.33 at 20 neurons (0.82 at 203), so the apical rule's `x_som - theta`
  term is negative on nearly every time step, the apical weights decay to zero instead of growing, the gain stays at 1
  and the gate multiplies the basal update by zero. A faster TD rate raises the TD error at texture time but also
  speeds the decay; only a low `theta_0` (0.25 to 1) lets the apical weights grow, and then the gain rises on every
  neuron alike (no texture selectivity), the total basal weight grows and the value estimator diverges (best of 192
  settings: 1 of 5 seeds at 1671 trials). Gated rules cannot self-start where no neuron is already texture-selective,
  so the shared environment for all rules has to be 100 to 203 neurons at `init_power` 9 to 12, where calcium works.
- Per-rule optimization at the shared environment (203 neurons, `init_power` 12, 30 distractors; sweeps `gh_mid`,
  `gh_comp`, `gh_fine`, `us_opt`, `us_fine`, `burst_opt`, `burst_fine`, `ctrl_fine`, `*_inh`; all in the `tdd` env,
  5 seeds). Learning rates dominate everything: the rule-free control goes from 1 of 5 seeds (all rates `LR`) to
  2 of 5 (`lr_policy` 0.064) to 5 of 5 at 1108 trials (`lr_ap` 0.032, `lr_policy` 0.128, `lr_td` 0.032). These
  are the two reference control conditions, "moderate" and "fast" rates (`ctrl_fine`, `ctrl_inh`, `ctrl_inh_fast`);
  under apical inhibition the fast control stays at chance and reaches expert 1268 trials after the lift in 5 of 5
  seeds, the moderate one 1398 in 2 of 5. `lr_td`
  0.064 diverges for every rule; `lr_bas` above 0.002 hurts every gated rule.
  - `gated_hebb` and `burst` add nothing beyond the rates: at moderate rates (`lr_ap` 0.016, `lr_policy` 0.064,
    `lr_td` 0.016) their best is 3 of 5 at about 1300 to 1350 trials vs the control's 2 of 5 at 1358; at the fast
    rates they reach 5 of 5 at 1072 to 1077 only with `lr_bas` 0.0005, where they barely change the weights, and the
    control does the same at 1108. Their constants are irrelevant: `gate_power` 0.5 to 4, `gh_theta_scale` 0.5 to 16,
    `gh_post` som or bas (two new `gated_hebb` knobs in `sandbox/rules.py` that scale the basal threshold or compare
    the un-gained basal drive with it), `burst_tau` 5 to 20, `burst_kappa` 0 to 0.2 all give the same result. Every
    setting that makes them form more texture detectors (`lr_bas` 0.004 or more, low `theta_0`) diverges. Under
    inhibition they diverge late in 2 of 5 seeds at `lr_bas` 0.0005 and in all seeds at 0.001 (gated Hebb).
  - `us` is the one gated rule with a real effect: with L1 normalization, `lr_bas` 0.001, `us_tau` 5 or 20 (no
    difference), `lr_ap` 0.016, `lr_policy` 0.064, `lr_td` 0.016 it reaches 5 of 5 at 1498 trials where the control
    with the same rates gets 2 of 5, and under inhibition it stays at chance and reaches expert 1612 trials after the
    lift in 5 of 5 seeds without divergence (control: 2 of 5, 1398 after the lift). Its good range is `lr_bas` 0.0005
    to 0.001 with `lr_td` 0.016 (0.032 works only without inhibition; 0.064 diverges) and `lr_policy` 0.064 to 0.128.
    It forms no texture detectors; the weights only reshape mildly (texture selectivity 0.88 to 0.5).
  - Reference: `calcium` (L1, `lr_bas` 0.004, `ca_ltd_frac` 0.3, `ca_theta_p` 1.0) at the same environment: 5 of 5 at
    780 to 793 trials with all rates at `LR`, 495 after the lift.
- Protocol for measuring each rule's improvement over the control (sweeps `f_*`, `f2_*` to `f5_*` in
  `sandbox/results`): environment fixed (203 neurons, `init_power` 12), policy and TD rates fixed at the fast
  condition (`lr_policy` 0.128, `lr_td` 0.032), only `lr_ap`, `lr_bas` and the rule constants optimized, first
  without inhibition (1800 trials), then the converged set under inhibition. Results at the fast condition:
  - Control: best `lr_ap` 0.064, 972 trials (10 seeds; 1033 at 5), stable over 4000 trials, 1223 after the lift.
  - `calcium` (L1, `ca_ltd_frac` 0.5, `ca_theta_p` 1.5, `ca_beta` 2, `lr_ap` 0.064; note these constants differ
    from the ones that were best at `LR` rates, and the old set `ca_ltd_frac` 0.3, `ca_theta_p` 1.0, `ca_beta` 0
    diverges in every seed at this TD rate): `lr_bas` 0.008 gives 534 trials (10 of 10; 45 percent faster than the
    control) and stays expert to 4000 trials, but under inhibition only 3 of 5 seeds hold (273 after the lift; the
    other 2 learn and then collapse to chance with 0 texture detectors, because 1800 blind trials of sharpening
    leave about one texture detector per seed). `lr_bas` 0.006: 573 (10 of 10), inhibition 4 of 5 (1 divergence,
    295 after the lift). `lr_bas` 0.004: 634 (9 of 10, 1 divergence), 2 of 5 diverge over 4000 trials without
    inhibition, but inhibition 5 of 5 with 328 after the lift. `lr_bas` 0.002: inhibition 5 of 5, 471 after the
    lift. Calcium therefore roughly halves the learning time at this condition, with a 10 to 20 percent per-seed
    risk of divergence or post-lift collapse in the 4000-trial protocols that no single `lr_bas` removes.
  - `bcm` (soft bound, `bcm_tau` 100, `bcm_e0_scale` 0.5, `lr_ap` 0.032): `lr_bas` 0.064 gives 548 (10 of 10),
    stable to 4000 trials, inhibition 3 of 5 (354 after the lift, 2 collapse); `lr_bas` 0.128 gives 408 (9 of 10,
    1 divergence), inhibition 4 of 5. The optimum sits on a ridge: `bcm_tau` 200 or `bcm_e0_scale` 0.25 fail in 3
    to 5 seeds and `bcm_tau` 50 with `bcm_e0_scale` 0.5 fails in 4.
  - `us` (L1, `lr_bas` 0.0005, `us_tau` 5, `lr_ap` 0.064): 909 (10 of 10), 6.5 percent faster than the control;
    inhibition 4 of 5 (1 divergence), 1246 after the lift.
  - `gated_hebb` (soft, `lr_bas` 0.0005, `gate_power` 1, `lr_ap` 0.064): 939 (3.4 percent faster); inhibition 2 of
    5 (3 divergences). `burst` (L1, `lr_bas` 0.0005, `burst_kappa` 0.2, `lr_ap` 0.064): 945 (2.8 percent faster);
    inhibition 5 of 5, 1204 after the lift. For all three gated rules `lr_bas` 0.00025 to 0.0005 is the whole
    usable range; 0.001 already loses seeds to divergence and the rule constants make no difference.
  - Same protocol at the moderate condition (`lr_policy` 0.064, `lr_td` 0.016; sweeps `m_*`, `m2_*` to `m4_*`):
    - Control: best `lr_ap` 0.128 to 0.256, 7 of 10 seeds at 1241 trials (936 at 0.256 for the 2 of 5 that learn);
      inhibition 2 of 5, 1084 after the lift.
    - `calcium` (L1, `ca_ltd_frac` 0.3, `ca_theta_p` 1.5, `ca_beta` 0; the LTD window that was best at `LR` rates,
      unlike the fast condition): `lr_bas` 0.032, `lr_ap` 0.256 gives 328 trials (10 of 10) and, unlike at the fast
      condition, passes inhibition in 5 of 5 seeds with 237 after the lift and no collapse; `lr_bas` 0.016 / `lr_ap`
      0.128 gives 404 (10 of 10), inhibition 5 of 5 at 254. Gains flatten beyond that (`lr_bas` 0.064, `lr_ap` 0.512:
      307, but final performance starts to drop). During inhibition performance sits slightly above chance (0.53 to
      0.54), i.e. the sharpened basal pathway alone lets the policy pick up a little.
    - `bcm` (soft, `bcm_tau` 100, `bcm_e0_scale` 0.5): `lr_bas` 0.256, `lr_ap` 0.128 gives 325 (10 of 10),
      inhibition 5 of 5 at 311; `lr_bas` 0.128, `lr_ap` 0.064 gives 447 (10 of 10), inhibition 5 of 5 at 355.
      `lr_bas` 0.512 diverges in 3 of 5. Total basal weight per neuron drops to about 0.75.
    - `us` (L1, `lr_bas` 0.001, `us_tau` 5, `lr_ap` 0.064): 1235 in 10 of 10 (control: 7 of 10 at 1241), i.e. it
      rescues the seeds the control loses without speeding up the others; inhibition 3 of 5, 1861 after the lift.
    - `gated_hebb` (soft, `lr_bas` 0.002, `gate_power` 2, `lr_ap` 0.128): 1224 in 9 of 10; inhibition 2 of 5 (3
      divergences). `burst` (soft, `lr_bas` 0.001, `burst_kappa` 0.2, `lr_ap` 0.128): 1217 in 9 of 10; inhibition
      3 of 5 (2 divergences). `lr_bas` 0.004 (gated Hebb) or 0.002 (burst) already diverges without inhibition.
    - Summary across both conditions: the apical-blind rules (calcium, bcm) cut the learning time by 45 to 75
      percent, and at the moderate condition they also survive the inhibition protocol; at the fast condition their
      long-horizon robustness is the limiting factor. The gated rules never beat the control's speed by more than 3
      to 7 percent; their contribution is to rescue seeds the control loses (`us` most reliably), and under
      inhibition they mostly diverge except `burst` at the fast condition.
- How to continue: the sandbox knobs added in this pass are `lr_ap`, `lr_policy`, `lr_td`, `lr_trace` (all
  default `LR`) in `simulate.py` and `gh_post`, `gh_theta_scale` for `gated_hebb` in `rules.py`. Finalist settings per
  condition are in `sandbox/results/finalists.json` (fast) and `m_finalists.json` / `m4_best.json` (moderate); their
  10-seed and inhibition pickles (`f3_*`, `m3_*`, `m4_*`) are kept, all other sweeps only as `summary.csv`
  (`python -m sandbox.tab <name>` prints them). Nothing plots the bottom-up results in the manuscript panels yet, the
  main-code rule (`hebb`, soft bound) is still the no-op-equivalent stale one, and `results/results_bup.pickle` is
  still stale. Natural next steps: pick the rule and condition to carry into the main code (calcium at the moderate
  condition is the strongest candidate), port its constants into `helper.py` / `simulate_seed()`, regenerate
  `results_bup.pickle` and the Matlab performance traces, and design the panels.
- Why `us` and `burst` underperform (one-seed diagnostics `sandbox/diag_gated.py`, `sandbox/diag_gated_ablation.py`,
  fast finalists): their slow variables `g_bar` / `p_bar` are correct cross-trial moving averages (rule `state` is
  created once per seed and updated every time step including the outcome step), so a per-trial reset is not the
  problem. Findings: (1) the post-synaptic factor at texture time is identical on T1 and T2 trials (difference 0.005
  vs mean 0.9) because the apical input `w_ap * x_pre_t[t]` is a shared timing signal; the only texture-specific
  factor is `x_bas` itself, so the rules are a multiplicative rich-get-richer on existing selectivity (the user
  considers this acceptable: the goal is textures vs distractors, not T1 vs T2). (2) `BKG` in `x_bas` boosts the
  weak texture weight relatively more, so selectivity erodes (top-20 neurons 0.92 to 0.73 for `us` under L1). (3)
  The outcome-step gain (up to 10 on all neurons) lifts `g_bar` to about 1.4, so for the first 300 trials the factor
  at texture time is negative on every neuron; but the ablation that ignores the outcome step does not speed
  learning (991 vs 1030 median, 5 seeds) because the inflated baseline acts as a threshold restricting potentiation
  to the 16 to 23 percent of neurons with high apical weight, and without it every neuron is potentiated alike. The
  other rules are immune because no input is active at the outcome step and they have no apical slow variable
  (`bcm`'s threshold sees `x_som` at most 0.1 there). (4) The tone-time gain equals the texture-time gain (2.27)
  but the tone occurs on every trial, so under competition the tone wins: with L1 the rules form 0 texture
  detectors at any rate and 10 tone detectors at `lr_bas` 0.002; 65 percent (`us`) / 51 percent (`burst`) of the
  update mass lands on distractor synapses. Net: they only re-weight neurons the apical loop already selected, so
  the speed-up is at most a few percent, and any rate high enough to move more weight raises the population drive
  and diverges.
- `hebb` with its own basal threshold (sweeps `hebb_tied`, `hebb_theta`, `hebb_theta2`, `hebb_theta_ctrl`,
  `hebb_theta10`, `hebb_theta_inh`; sandbox knob `theta_bas_0`, default `None` = apical thresholds): with the
  apical rule's own hyperparameters (`lr_bas` = `lr_ap`, soft bound, shared `theta_0` 4) the rule diverges in every
  seed at every rate condition (trial 190 to 700), because `x_som - theta` is positive on every driven step and all
  active synapses potentiate monotonically to 1. Raising only the basal threshold to `theta_bas_0` 16 (1.5 times the
  median neuron's strongest un-gained response, the calcium rule's LTP setting) turns it into a pruning rule: within
  300 trials the total weight per neuron falls from 1 to about 0.25, tone weights go to 0, 40 to 60 of 203 neurons
  keep any input, 30 distractor detectors form apically blind (the neurons whose strongest input crosses un-gained)
  and 3 to 8 texture detectors form through the apical gain. Learning is then faster than the control at tied rates
  in 10 of 10 seeds: fast condition 553 trials (`theta_0` 1) or 642 (`theta_0` 4) vs control 1033; moderate 722 or
  879 vs 1235 (7 of 10). Under apical inhibition it stays at chance (0.505 to 0.52) and reaches expert 597 to 923
  trials after the lift, i.e. the same time as without inhibition, so unlike calcium it satisfies both inhibition
  criteria. The window is narrow: `theta_bas_0` 8 to 10 diverges, 12 is borderline (4 of 5 diverge, the rest learn in
  about 480), 20 to 24 loses 2 to 5 seeds because too few texture synapses cross even with gain; the window scales
  with `init_power` through the strongest-response ratio. The apical threshold matters too: `theta_0` 1 beats 4 for
  the rule and the control at moderate and `LR` rates, `theta_0` 16 stops all learning, and at the fast condition
  the control with `theta_0` 1 diverges in 5 of 5 seeds while the pruned model is stable. Pickles are kept only for
  `hebb_theta10` and `hebb_theta_inh`.
- `oja` (sweeps `oja_f`, `oja_f2` at the fast condition, `oja_m` at the moderate one; `bound` clamp and none are
  bit-identical because the weights never leave [0, 1]; `oja_decay` 1 to 64, `lr_bas` 0.0001 to 0.008; summaries
  only, pickles deleted): never better than the control. Textbook decay 1 diverges in every seed, decay 4 in most
  (at `lr_bas` 0.002 or more), decay 16 or 64 keeps the drive bounded but leaves performance at chance in every
  seed. Only the settings that barely act stay usable: at `lr_bas` 0.0001, decay 2, 5 of 5 seeds at 1159 trials
  (fast; control 1033), and at the moderate condition 2 of 5 (control 3 of 5). Mechanism (one-seed weight
  trajectories): Oja's fixed point is `w_i` proportional to `E[x_i x_som]`, so with sparse binary inputs the weight
  vector converges to the input-frequency profile (tone 1, distractors up to 0.97, textures 0.5): the weights
  homogenize (median strongest weight 0.30 to 0.09, texture selectivity 0.88 to 0.04 to 0.4), the tone becomes the
  strongest input on 50 to 75 percent of neurons, and the L1 drive grows (to 1.4 to 1.9 at decay 4) while the L2
  norm shrinks toward `1 / sqrt(decay * gain)`. Because the apical gain is a timing signal shared by tone and
  texture, it cannot bias the competition toward the textures. Why it fails where `hebb` with `theta_bas_0` 16
  works: `hebb` has an absolute per-neuron threshold, so it depresses every active synapse on every step unless the
  neuron's response already exceeds the threshold (pruning that shrinks the drive and keeps the initial winners,
  with the apical gain lifting texture synapses over it); Oja has no threshold, potentiates every active synapse
  however weak, holds strong ones down with a decay proportional to their own weight, and its single-synapse
  equilibrium falls with the gain, so it erases the initial selectivity instead of amplifying it. Todo item (2)
  is closed with this result; `oja_decay` was added to the columns of `sandbox/tab.py`.
- `gated_hebb` diagnosed (todo item 1; sweeps `gh_tb`, `gh_tb2` at the fast condition, summaries only; one-seed
  diagnostics `sandbox/diag_gh.py`, output in `sandbox/results/diag_gh_lr*.txt`): a raised basal threshold does not
  rescue it. With `gh_post` som (the gained `x_som` compared with the threshold, as in `hebb`) every setting of
  `theta_bas_0` 8 to 64, `lr_bas` 0.002 to 0.256 and `theta_0` 1 or 4 diverges in 5 of 5 seeds (a few seeds survive
  only at `theta_bas_0` 64), with 12 to 35 texture detectors, up to 164 distractor detectors and `w_sum` up to 3.2;
  unlike `hebb` it never prunes, because depression can only occur while the gate is open (tone and texture time),
  so the distractor synapses active at all other steps are never touched, and at the gated steps the gain (up to
  10) pushes many neurons above any threshold. With `gh_post` bas (un-gained drive compared with the threshold)
  nothing diverges, but the rule is then pure depression: no texture detectors form, the task weight fraction falls
  from 0.094 to 0.065 to 0.075, and learning is the control's (1035 vs 1033 trials at `theta_bas_0` 16; worse at
  32 and 64, where 1 to 5 seeds fail). Diagnostics at the fast finalist (`lr_bas` 0.0005, soft, seed 0, expert 784)
  and at `lr_bas` 0.004 (diverges at trial 792): (1) the rule is almost purely potentiating (depression is about 1
  percent of the update mass); (2) the factor at texture time is positive on about 20 percent of the neurons from
  the start (those whose gained response already exceeds theta) and its T1 minus T2 difference per neuron is 0.02
  against a mean of 0.3, so as for `us` / `burst` it only re-weights neurons the apical loop selected; (3) 48 to 61
  percent of the potentiation lands on distractor synapses that happen to be active at the gated steps, 36 to 40 on
  textures, 3 to 11 on the tone (small tone weights grow slowly under the soft bound), so distractor detectors form
  alongside texture detectors (12 and 15 at `lr_bas` 0.004); (4) at `lr_bas` 0.004 the fraction of neurons with a
  positive factor rises from 20 to 45 percent by trials 600 to 900 (potentiation raises `x_som`, more neurons cross
  theta, the apical loop strengthens) and the value estimator diverges. Texture selectivity is untouched under the
  soft bound (0.92 to 0.93 on the top-20 neurons), so the `BKG` erosion seen for `us` under L1 does not apply.
  `theta_bas_0` was added to the columns of `sandbox/tab.py`.
- Why `us` and `burst` fail where `hebb` with a basal threshold, `calcium` and `bcm` work (2026-09-29; one-seed
  diagnostics `sandbox/diag_compare.py`, seeds 0 and 1, fast condition finalists, outputs in
  `sandbox/results/diag_compare/`; sweep `gate_us_burst`, summary only; ablation `sandbox/diag_tone_ablation.py`):
  - The working rules are pruners. Depression is 94 to 100 percent of `hebb`'s raw update mass, 56 to 92 percent
    of `calcium`'s and 95 to 99 percent of `bcm`'s, most of it on distractor synapses at distractor-only steps,
    and the update sign is a function of the neuron's own drive relative to a fixed per-neuron threshold, applied
    on every step: only 1 to 10 percent of the neurons potentiate at any step, potentiation is concentrated (80
    percent of the texture-row potentiation on 6 to 23 neurons), and about half of it lands on the neuron's
    currently strongest synapse. `hebb` drives the total weight per neuron to about 0.25 and the tone weights to
    exactly 0 within 300 trials; the apical gain then lifts the texture response of the apically selected neurons
    over the threshold (potentiation correlates with the apical weight, rho 0.6), so 3 to 8 texture detectors form
    while everything else is pruned and the population drive stays bounded.
  - The gated rules are potentiators whose sign is set by the apical state, not by the neuron's own drive:
    potentiation is 65 to 99 percent of their mass (`us` 0.65 to 0.75, `burst` 0.98, `gated_hebb` 0.99), it only
    happens at tone and texture steps, on 16 to 37 percent of the neurons (the ones with high apical weight, rho
    0.7 to 0.99), and at the texture step it is split about equally between the texture row and whatever
    distractors happen to be co-active (0.45 to 0.52 vs 0.40 to 0.48), because every active synapse of a
    potentiated neuron gets the same increment. Their depression at distractor-only steps is proportional to the
    neuron's drive (`us`: Spearman rho between the post-synaptic factor and `x_bas` is minus 0.9), i.e. the most
    strongly driven neurons lose the most, the opposite of a winner-take-all, and it is too weak to prune (total
    weight and median strongest weight stay at their initial values). Result: no texture detectors (0 to 1), the
    same expert trial as the control on the same seed (774 to 784 vs 803), and one third instead of one half of
    the potentiation on the strongest synapse.
  - The tone is what diverges. Because the apical gain is the same timing signal at tone and texture time and the
    tone occurs on every trial, the tone's share of the gated rules' potentiation grows from 0 to 0.3 during
    learning (as the TD error moves to the tone step) and the tone weight grows on every apically selected neuron
    (`us` 0.03 to 0.06 at `lr_bas` 0.0005; 7 to 15 tone detectors at 0.004 to 0.008), which raises the population
    drive at the tone step and blows up the value estimator. Causal test: zeroing the tone row's basal update
    removes every divergence at `lr_bas` 0.004 (`us` 0 of 5 diverge, 829 trials; `burst` 0 of 5, 931) and for
    `burst` also at 0.008 (910), but learning is still not faster than the control (1033) because the rules still
    form only 0 to 4 texture detectors. In `hebb` the tone never crosses the threshold even with gain (tone weights
    are small on every neuron) and is pruned to 0.
  - A per-neuron response threshold on the gated rules does not help (sweep `gate_us_burst`; sandbox knob
    `theta_gate` = `pot` restricts potentiation to neurons with `x_som` above the basal threshold, `all` switches
    the basal plasticity of sub-threshold neurons off; `theta_bas_0` 8 to 24, `lr_bas` 0.0005 to 0.032, both
    bounds, 5 seeds): the gate barely binds, because the gated rules already potentiate only where the gain is
    high and the gained response of those neurons exceeds every tested threshold (with the soft bound the results
    are identical for all three thresholds). Best setting `us` L1 `lr_bas` 0.002 `theta_bas_0` 24 `pot`: 823 trials
    in 5 of 5 (ungated 909); everything at `lr_bas` 0.008 or more still diverges or fails, via tone detectors.
    Todo item (5) is closed with this result.
  - Summary: what makes a rule work in this model is a post-synaptic factor that is negative for every synapse
    active on a sub-threshold neuron (constant pruning of distractors and of the tone) and positive only when
    the neuron's own gained response crosses a per-neuron threshold (winner-take-all across and within neurons,
    with the apical gain deciding which texture neurons cross). The apically gated rules have neither: their sign
    is shared across the population and across tone and texture, and their magnitude scales with the drive, so
    they re-weight the apically selected neurons uniformly, cannot prune, and any rate that moves real weight
    grows the tone drive until the value estimator diverges. The gated rules would need a subtractive, drive-
    independent depression term (as in `hebb`) to be viable; that is a rule-design decision for the user.
- Open decision: the shared environment can be 203 neurons at `init_power` 12 (100 neurons also works for calcium but
  not for any gated rule). Calcium and `hebb` with `theta_bas_0` 16 are the two rules that beat the control by a wide
  margin (calcium 328 to 534 trials, `hebb` 553 to 722, control 1033 to 1235); only `hebb` passes both inhibition
  criteria without post-lift collapse. `us` gives a modest, clean improvement; `gated_hebb` and `burst` are
  indistinguishable from tuning the learning rates. Still open: accept apical-blind sharpening or pruning, or add
  a population-level competition so that a gated rule can concentrate potentiation on few neurons.
- Todo list (agreed with the user on 2026-09-25, in this order): (1) diagnose `gated_hebb` as deeply as `us` /
  `burst` (done on 2026-09-29, see the `gated_hebb` bullet: a raised basal threshold does not help); (2) explore `oja` (done on
  2026-09-28, see the `oja` bullet: no usable regime); (3) a normalization step that keeps the
  bottom-up weights from growing too large; (4) sparse weight initialization so selectivity is structurally
  predefined (`init_power` does this; combine with a normalization, since the control at `init_power` 24
  diverges); (5) a basal threshold under which no plasticity occurs, tried for `hebb`, not yet for `us` / `burst`
  where `BKG` erodes selectivity (done on 2026-09-29, see the `us` / `burst` bullet: a threshold gate does not
  bind). Constraints the user set: every mechanism must be biologically plausible (no
  time-step-specific baselines, no excluding the outcome step from a running average; such things are allowed only
  as one-off causal ablations), and every explored parameter must be a sandbox knob with the main-code value as
  default. After that: pick the rule and condition for the main code (candidates: calcium at the moderate condition,
  `hebb` with `theta_bas_0` 16 at either condition), port it into `helper.py` / `simulate_seed()`, regenerate
  `results_bup.pickle` and the Matlab performance traces, and design the panels. The `theta_bas_0` knob, the three
  `diag_gated*` / `diag_gh` scripts, the `theta_bas_0` and `oja_decay` columns in `sandbox/tab.py`, the README rows
  and these notes were added on 2026-09-25 to 29 and committed as ae1371b on 2026-09-29 (not pushed). Delete pickles of new
  exploratory sweeps once `summary.csv` exists (the disk was at 97 percent, 79 after this cleanup).
