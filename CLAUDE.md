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
- `calcium` (LTD/LTP thresholds) with L1 normalization at `lr_bas` 0.002 to 0.004 is the strongest rule: 5 of 5 seeds expert, 780 to 1100 trials, no divergence, robust to its threshold parameters as long as the LTD window is wide (`ca_ltd_frac` 0.3 to 0.5). Mechanism: a within-neuron winner-take-all that turns nearly every neuron into a one-hot detector of its dominant input (about 170 distractor and 10 to 20 texture detectors); the explicit apical term of the calcium proxy (`ca_beta`) makes no difference (the gain still enters through `x_som`). `bcm` with the soft bound at 0.064 does the same faster (637 trials) but diverges in 2 of 5 seeds.
- Because that sharpening does not need the gain (the rule reads `x_som`, but at gain 1 it still sharpens), it also runs during apical inhibition: performance stays at chance (criterion 1 holds for every rule tested), but after the lift learning is faster than without inhibition (calcium 789 vs 1107 trials, bcm 477 vs 637), so criterion 2 (same time after the lift) is violated in that direction.
- The rules with an explicit apical factor (`burst`, `gated_hebb`, `us`) are silent during inhibition and build texture detectors only once the apical signal exists, so they satisfy both criteria by construction, but they cannot bootstrap learning: at `lr_bas` 0.004 with the soft bound they turn 1 in 5 control seeds into 2 to 3 of 5 (1415 to 1607 trials), the detectors they form are mostly texture detectors (about 10 to 17 vs 8 to 15 distractor), and with a higher rate or the L1 bound the shared apical signal spreads potentiation over the whole population (task fraction rises, no neuron becomes a detector) and the value estimator diverges, in the 4000-trial protocol also late after learning.
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
    - Summary across both conditions: the rules without an explicit apical factor (calcium, bcm) cut the learning time by 45 to 75
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
  keep any input, 30 distractor detectors form without the gain (the neurons whose strongest input crosses un-gained)
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
- Why `us` and `burst` fail where `hebb` with a basal threshold, `calcium` and `bcm` work (2026-09-29; terminology
  corrected by the user the same day: these three rules read `x_som = gain * x_bas`, so the apical signal modulates
  them through the somatic rate; they are not apical-blind, they only lack an explicit apical factor; the constant
  LTD term below is the one apical-independent rule; one-seed
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
  - The rules with an explicit apical factor are potentiators whose sign is set by the apical deviation, not by the
    neuron's own gain-modulated drive:
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
    with the apical gain, acting through the somatic rate, deciding which texture neurons cross). The rules with an
    explicit apical factor have neither: their sign
    is shared across the population and across tone and texture, and their magnitude scales with the drive, so
    they re-weight the apically selected neurons uniformly, cannot prune, and any rate that moves real weight
    grows the tone drive until the value estimator diverges. The gated rules would need a subtractive, drive-
    independent depression term (as in `hebb`) to be viable; that is a rule-design decision for the user.
- Subtractive depression added to `us` and `burst` (2026-09-29, sandbox knobs `lr_dep` and `dep_form` in
  `simulate.py`, default off; sweeps `dep_us_burst`, `dep_none`, `dep_inh`, `dep_mod`, summaries only; fast
  condition unless stated, `theta_bas_0` 16, 5 seeds): the term is added to any rule's update, `thr` is the
  depression half of `hebb` (`- lr_dep * x_in * relu(theta_bas - x_som)`), `const` depresses every active synapse
  by `lr_dep`; with `rule = none` it is a pruning-only baseline.
  - It helps, but the gated potentiation contributes nothing: for every bound, form and rate the numbers of `us`,
    `burst` and the rule-free pruning baseline are identical to within a few trials (`us` / `burst` at `lr_bas`
    0.0005 to 0.008 versus `none`: 398 / 396 / 397 for `const` L1 0.016; 896 / 885 / 895 for `thr` soft 0.016).
  - `const` with the L1 bound is the fastest setting found in the whole project: 392 to 399 trials in 5 of 5
    seeds at `lr_dep` 0.016 (control 1033, `hebb` 553, `calcium` 534, `bcm` 548), 410 to 439 at 0.064, 738 at
    0.256; at the moderate condition 495 (`lr_ap` 0.128) to 547 (`lr_ap` 0.064) versus the control's 1241 in
    7 of 10 (`calcium` 328). Mechanism: each activation subtracts a constant from the synapse and L1
    renormalization hands the weight to the synapses that fire least, so every neuron ends as a one-hot detector
    of its input with the largest initial weight-to-frequency ratio (196 distractor, 6 texture, 1 tone detectors;
    task weight fraction 0.03, `w_bas` fixed once one-hot). It reads neither `x_som` nor the apical activation, so it is the one genuinely
    apical-independent rule tested: under inhibition
    performance sits at 0.54 and expert comes 325 trials after the lift (397 without inhibition), the same
    criterion 2 asymmetry as `calcium`, and it stays expert to 4000 trials without divergence.
  - `thr` with the soft bound (the `hebb` depression alone) reaches 833 to 912 trials in only 3 of 5 seeds: without
    `hebb`'s strong potentiation on above-threshold neurons the weights decay to `w_sum` 0.03 to 0.2 and the
    population loses drive (final performance 0.7 to 0.8); the gated rules' potentiation at `lr_bas` up to 0.008
    is too weak to hold the detectors (`hebb` uses 0.064). `thr` with L1 diverges in every seed, including the
    pruning-only baseline, because the gained tone and texture responses cross the threshold and are spared while
    everything else is depressed and renormalized away, so 3 to 4 tone and 12 to 20 texture detectors form and
    the tone drive explodes. `const` with the soft bound decays every weight to 0 (chance).
  - Conclusion: the speed-up of every rule that works here (`hebb` with a basal threshold, `calcium`, `bcm`, and
    now plain constant LTD with L1) comes with sharpening of the basal weights into one-hot detectors, which even
    an apical-independent rule with no potentiation delivers, and the explicit apical potentiation of `us` and
    `burst` is irrelevant at every rate at which it does not diverge. How much of the `hebb` / `calcium` / `bcm`
    effect enters through the gain in `x_som` is measured with the `rule_sees_gain` ablation (next bullet).
- How much of the `hebb` / `calcium` / `bcm` effect enters through the apically modulated somatic rate (2026-09-29,
  after the user's correction; sandbox knob `rule_sees_gain`, default `True`; `False` hands the rule `x_som`
  computed at gain 1 and resting apical variables while the network keeps the real gain; sweep `gain_ablation`,
  fast finalists, 5 seeds, summary only): with the gain `calcium` 548, `bcm` 558, `hebb` 587 trials (control 1033);
  without it `calcium` 613 (5 of 5), `hebb` 758 (5 of 5), `bcm` 542 but 2 of 5 diverge (its sliding threshold no
  longer tracks the gained rate, so potentiation runs away into 3 tone and 11 texture detectors). So the implicit
  apical route carries about 40 percent of `hebb`'s advantage over the control, about 15 percent of `calcium`'s,
  and is what keeps `bcm` stable; the rest is sharpening that also happens at gain 1 (which is why these rules keep
  sharpening during apical inhibition). The apical-independent constant LTD (397) is faster than any of them with
  or without the gain, so in this model the sharpening, not the apical modulation of it, sets the learning speed.
- New goal (user, 2026-09-29): a manuscript figure showing that apical top-down input can support bottom-up
  plasticity that speeds learning beyond the apical loop alone, with `calcium`, `hebb` (basal threshold) and `bcm`
  (`us` / `burst` set aside). Two steps: (1) control regimes (no basal plasticity) that learn in about 1000 trials
  without inhibition and after the lift, with as many distractors as possible (towards 200) and the densest init
  possible (lowest `init_power`), all learning rates free, as many distinct regimes as can be found; (2) per
  regime, add the three rules and tune them to cut the time, keeping the no-inhibition and post-lift times similar.
- Step 1 results (sweeps `env1_screen` to `env7_nz`, summaries only; `n_z` 203, `theta_0` 4, `lr_ap` 0.064,
  `lr_td` 0.032, `lr_policy` 0.256 unless stated; 10 seeds without inhibition, 5 seeds in the inhibition protocol;
  `n_noise` and `n_trials` were added to the columns of `sandbox/tab.py`):
  - Dense inits never learn: at `init_power` 1 to 6 every setting of `n_noise` 30 to 200, `theta_0` 0.25 to 4,
    `lr_ap` 0.064 to 0.256 and `lr_policy` 0.128 to 0.256 stays at chance for 1800 to 3000 trials. One-seed
    diagnostics (`sandbox/results/diag_compare/env_*.txt`): with `theta_0` 0.25 the apical loop does start (apical
    weights 0.1 to 0.5, TD error moves to tone and texture time) but the policy stays at chance, because no neuron
    carries a texture-specific response that stands out from the common-mode drive; lowering `theta_0` only raises
    the gain on every neuron alike. Rates never rescue a dense init.
  - The required sparsity scales with the number of stimuli: the control learns from `init_power` about 0.4 to 1
    times `n_noise` + 3 upwards (power 8 to 12 at 30 distractors, 32 to 48 at 60, 64 to 128 at 100, 192 at 150),
    and from there higher powers are faster but at 30 distractors powers 24 to 48 diverge (several tone
    detectors). `lr_td` 0.032 beats 0.016 everywhere, `lr_policy` 0.256 is 5 to 10 percent faster than 0.128,
    `lr_ap` 0.032 to 0.128 makes little difference.
  - Regimes at about 1000 trials (expert proxy, seeds reaching expert; after the lift in the inhibition protocol):
    30 distractors, power 12: 915 (10 of 10), lift 1143 (5 of 5). 60 distractors, power 32: 1067 (8 of 10), lift
    1190 (4 of 5); power 48: 655 (9 of 10), lift 633 (4 of 5). 100 distractors, power 96: 969 (9 of 10), lift
    1001 (4 of 5); power 128: 765 (9 of 10), lift 784 (4 of 5); power 64: 1134 (6 of 10). 150 distractors, power
    192: 873 (6 of 10), lift 939 (3 of 5). 200 distractors, power 192: 1048 (3 of 10), lift 948 (2 of 5); the
    identity init (the paper's default model) at these rates: 669 (3 of 3). In every regime the post-lift time
    matches the no-inhibition time within about 20 percent, and performance during inhibition is 0.51 to 0.53.
  - The seeds that fail are a texture-coverage lottery, not a rate problem: the random init gives each stimulus
    about 203 / (`n_noise` + 3) dominant neurons, so from 60 distractors upwards some seeds have no neuron whose
    strongest input is T1 (or T2). Verified per seed with `get_params_bu`: seed 4 has 0 T1-dominant neurons at
    (60, 48) and (100, 96) and is the seed that fails at every rate; at 150 to 200 distractors 3 to 5 of 10 seeds
    lack a neuron for one texture. A larger population fixes this at 100 distractors (`n_z` 400, power 96: 873 in
    5 of 5; `n_z` 800: 894 in 5 of 5) but not at 200 (`n_z` 400 or 800, power 192: 1 to 2 of 5, because the
    hundreds of distractor detectors swamp the policy readout), and `n_z` 400 or 800 at power 192 with 100
    distractors diverges. `n_z` was not freed by the user; it is the lever if 100 or more distractors with every
    seed learning is wanted.
- Step 1 closed by the user (2026-09-29): `n_z` stays 203, the regime is about 30 distractors around `init_power`
  12, better-motivated init families may be proposed, and all control rates must be tuned. Fully tuned control
  (sweeps `c30_rates`, `c30_rates3`, `c30_place`, `s2_ctrl`, `s2_ctrl_inh`): `lr_policy` 0.256, `lr_td` 0.048
  (0.056 is 5 percent faster but at the divergence edge, 0.064 diverges), `lr_trace` 0.256 (0.016 is 25 percent
  slower; 0.512 to 1.0 no better), `lr_ap` 0.064 (0.128 equal). At these rates the control at 30 distractors learns
  in 597 trials at power 12 (10 of 10; 808 after the lift, 5 of 5) and 814 at power 10 (10 of 10; 1172 after the
  lift), so power 10 is the closer match to the 1000-trial target and both powers are carried. Power 8 loses 4 of
  10 seeds; 40 distractors fail at every power. Alternative inits (sandbox `init` = `lognormal` with `init_sigma`,
  `dirichlet` with `init_alpha`; `sweep c30_init`): lognormal sigma 2 has the same sparsity as power 12 (median
  strongest weight 0.32, about 13 texture-dominant neurons) but its heavier tail gives a few very strong neurons,
  so at tuned rates it diverges in 1 to 4 of 10 seeds (sigma 2 to 2.5); Dirichlet 0.05 learns in 474 (5 of 5) at
  the slower rates. Kept as options, the power init stays the working one.
- Step 2 at the tuned control rates (policy 0.256, TD 0.048, trace 0.256 fixed; `lr_ap`, `lr_bas` and the rule
  constants free; sweeps `s2_rules`, `s2_refine`, `s2_casilent`, `s2_fin10`, `s2_fin_inh`, `s2_hebb2`; 10 seeds
  without inhibition, 5 in the inhibition protocol; `sandbox/best.py` prints per-rule bests):
  - `calcium` (L1, `lr_bas` 0.001, `ca_ltd_frac` 0.5, `ca_theta_p` 1.5, `ca_gamma_d` 1.0, `ca_beta` 2, `lr_ap`
    0.128): power 12: 497 (10 of 10), 486 after the lift (5 of 5) versus the control's 597 and 808, i.e. the
    no-inhibition and post-lift times are now the same; power 10: 666 (10 of 10), 573 after the lift (`ca_gamma_d`
    0.5: 691 and 469) versus 814 and 1172. `lr_bas` 0.002 or more loses seeds to divergence at this TD rate. A
    narrow LTD window (`ca_ltd_frac` 0.8 to 1.0, `sweep s2_casilent`), which makes the rule nearly silent without
    gain, learns at the control's speed (603 at power 12, 891 at power 10) with about 10 distractor detectors, so
    the speed-up of calcium comes with the sharpening at gain 1.
  - `bcm` (soft, `bcm_e0_scale` 0.5): power 12 `lr_bas` 0.032, `bcm_tau` 70, `lr_ap` 0.064: 386 (10 of 10),
    inhibition 4 of 5 at 373 after the lift; power 10 `lr_bas` 0.064, `bcm_tau` 50: 429 (10 of 10) but under
    inhibition 3 of 5 seeds collapse after the lift (final performance 0.71). Faster `lr_bas` (0.096) diverges in
    3 of 10.
  - `hebb` (soft, `theta_0` 1): power 12 `lr_bas` 0.128, `theta_bas_0` 16, `lr_ap` 0.128: 340 but only 7 of 10
    (3 diverge), while the inhibition protocol passes 5 of 5 with 321 after the lift; `theta_bas_0` 18 with
    `lr_bas` 0.256: 425 (9 of 10), inhibition 4 of 5 at 378. Power 10: `theta_bas_0` 14 diverges in 4 of 10 (468
    after the lift, 5 of 5), 16 loses 2 of 10 to over-pruning (704; 772 after the lift). Per-seed check: the seed
    that diverges is the one whose init has the most texture-dominant neurons (18); it ends with 9 texture
    detectors and half the pruning of the other seeds (`w_sum` 0.51 vs 0.2), so the texture-time drive grows until
    the value estimator diverges. The bound (`l1`) and a lower `lr_ap` (0.032) do not remove it.
- Step 2 closed for now (sweeps `s2_bcm_inh`, `s2_hebb3`, `s2_hebb3_inh`, `s2_td32`, `s2_td32_inh`, `s2_ca32`,
  `s2_ca32_inh`, summaries only). Two rate conditions, both with `lr_policy` 0.256 and `lr_trace` 0.256, 30
  distractors, `n_z` 203; 10 seeds without inhibition, 5 in the inhibition protocol (numbers are median expert
  trial / median expert trial after the lift; all seeds reach expert unless stated):
  - TD 0.048 (the control's fully tuned rate): control power 10 814 / 1172, power 12 597 / 808. `calcium` (L1,
    `lr_bas` 0.001, `ca_ltd_frac` 0.5, `ca_theta_p` 1.5, `ca_gamma_d` 1.0, `ca_beta` 2, `lr_ap` 0.128): power 10
    666 / 573, power 12 497 / 486, all seeds. `hebb` (soft, `theta_0` 1): power 12 `theta_bas_0` 16, `lr_bas`
    0.128, `lr_ap` 0.128: 340 but 3 of 10 diverge, inhibition 5 of 5 at 321; `theta_bas_0` 18: 425 (9 of 10),
    inhibition 4 of 5 (one seed over-pruned during the 1800 blind trials); power 10 `theta_bas_0` 14: 4 of 10
    diverge, 15: 9 of 10 at 430 but inhibition 4 of 5, 16: 8 of 10 at 704. `bcm` (soft, `bcm_e0_scale` 0.5):
    power 10 `lr_bas` 0.064, `bcm_tau` 50: 429 (10 of 10) but 3 of 5 seeds collapse after the lift; power 12
    `lr_bas` 0.032, `bcm_tau` 70: 386 (10 of 10), 4 of 5 at 373; every other `lr_bas` / `bcm_tau` / `lr_ap`
    combination tried loses 1 to 3 of 5 seeds to the post-lift collapse.
  - TD 0.032: control power 10 1084 / 1390, power 12 713 / 967. `hebb` power 10 (`theta_bas_0` 14, `lr_bas`
    0.256, `lr_ap` 0.064): 417 / 534, all seeds; power 12 (`theta_bas_0` 16, `lr_bas` 0.128, `lr_ap` 0.128):
    343 / 365, all seeds; the post-lift to no-inhibition ratio (1.28 at power 10) equals the control's (1.28).
    `calcium` power 10 (`lr_bas` 0.001, `ca_gamma_d` 1.0): 847 / 760; (`lr_bas` 0.002, `ca_gamma_d` 0.5): 714 /
    326, i.e. at this TD rate the faster calcium settings are again asymmetric; power 12 (`lr_bas` 0.001,
    `ca_gamma_d` 1.0): 598 / 652. `bcm` still collapses after the lift in 2 to 3 of 5 seeds at both powers.
  - Reading: `hebb` with its own basal threshold is the rule that meets the figure's requirement (large speed-up,
    every seed, same no-inhibition and post-lift times as the control's ratio) but only at TD 0.032, where the
    control is 25 percent slower than at its own optimum; at TD 0.048 its threshold window between divergence
    (too many texture detectors in seeds with a rich texture init) and over-pruning during inhibition no longer
    contains every seed. `calcium` meets the requirement at TD 0.048 with a smaller speed-up (17 to 18 percent
    without inhibition, 40 to 50 percent after the lift). `bcm` is out: its sliding threshold cannot recover the
    texture weights it loses during 1800 trials at gain 1. Candidate figure settings: control at power 10, TD
    0.032 (1084 / 1390) versus `hebb` (417 / 534); or control at TD 0.048 (814 / 1172) versus `calcium` (666 /
    573). Next: pick, port the chosen rule and constants into `helper.py` / `simulate_seed()`, regenerate
    `results_bup.pickle` and the Matlab traces, and design the panels.
- Base condition revisited (user, 2026-09-29): the control itself must learn equally fast without inhibition and
  after the lift; the user's reading is that the policy gets corrupted during the 1800 inhibited trials and the
  remedy is a lower policy rate compensated by higher other rates. Sweeps `c30_pol`, `c30_pol_inh`, `c30_pol2`,
  `c30_pol2_inh`, `c30_reset`, `c30_reset256`, `c30_rng`, `c30_fresh`, `c30_cand`, `c30_cand2500`, `c30_cand_inh`
  (summaries only; `sandbox/pair.py` pairs a sweep with its inhibition twin; `reset_at_lift` ablation knob):
  - What changes during inhibition (one seed, policy 0.032): nothing but the policy. The TD-error trace `x_pre_t`
    is exactly 0 (the outcome-step update of the value estimator is multiplied by `x_som - BKG` = 0 and the
    estimator's weights stay 0), so the apical weights do not move at all; the policy weights drift to a norm of
    0.23 (they reach 0.78 when expert) because the policy update still runs at 1/10 strength with `x_som` = `BKG`.
  - The drift is real but small: at policy rate 0.256 (TD 0.048, apical 0.064, trace 0.256, power 12) the
    post-lift expert trial is 808 without reset, 766 with the policy reset at the lift (5 seeds), and resetting the
    apical or TD weights as well changes nothing more. With the random generators also reset (`all_rng`, 20
    seeds) the post-lift run reproduces the fresh run exactly (701 = 701), and with only the state reset it is 757:
    an 8 percent difference that is pure sampling noise of the 20-seed median (the post-lift slice is a different
    random realization). So of the 35 percent gap at policy 0.256 about 7 points are the policy drift and the
    rest is sampling scatter and the slow tail of the proxy; the user's remedy removes the drift part.
  - Paired results (20 seeds; no-inhibition horizon 2500 trials; `lost` = runs of 40 not reaching expert or
    diverging): power 12, policy 0.016, TD 0.056, apical 0.064: 1138 / 1140 (ratio 1.00, lost 2); policy 0.016,
    TD 0.064, apical 0.064: 1102 / 1096 (1.00, lost 4); policy 0.016, TD 0.064, apical 0.128: 1160 / 1172 (1.01,
    lost 3); policy 0.032, TD 0.064, apical 0.128: 909 / 936 (1.03, lost 2); policy 0.032, TD 0.056, apical
    0.064: 893 / 951 (1.07, lost 1); power 10, policy 0.032, TD 0.064, apical 0.064: 1142 / 1262 (1.11, lost 4).
    Policy 0.008 is too slow (1350 to 1600) and loses seeds; `lr_trace` 1.0 helps nothing; apical 0.256 at power
    10 diverges. Performance during inhibition is 0.50 everywhere. A 1800-trial horizon under-counts a control
    at about 1100 trials (slow seeds are marked not reached), so use 2500 trials for such controls.
  - Recommended base condition: 30 distractors, power 12, policy 0.016, TD 0.056, apical 0.064, trace 0.256
    (1138 without inhibition, 1140 after the lift), or policy 0.032, TD 0.064, apical 0.128 (909 / 936) if 900
    trials is close enough to the target. Rule tuning (step 2) has to be redone at the chosen condition.
- Figures without Matlab (2026-09-30): `sandbox/smith.py` is a validated port of the Smith et al. estimator
  (identical expert and learning trials to `Smith/performances.mat`, traces within 3e-3 = the Matlab Monte Carlo
  noise), `sandbox/figures.py` builds `results/figures/<name>/{default,perturbed}.pickle` for any sandbox
  configuration (10 seeds, 1800 and 4000 trials, `store_full`, `sort_neurons` so that neurons 0, 1, 2 are the best
  tone / T2 / T1 neurons, `gain_at_preferred` so that the gain record is the gain at each neuron's preferred input,
  `K_W_BAS` holding the initial weights so the selectivity panels bin by initial selectivity) and reproduces all
  manuscript panels except `fs12b_theta0` into `results/figures/<name>/panels/` (PDF, SVG, PNG).
  `sandbox/selectivity.py` adds a cross-configuration comparison (`compare_*.png`: basal selectivity histograms
  initial vs final, final vs initial weight per synapse class, late response selectivity vs initial basal
  selectivity, detector counts over trials) and `fs12cd_wide_*.png`, the fs12cd panel with a +-10 range (the
  manuscript's +-4 range and 0.1 responsiveness threshold hide the gained detectors of the rules).
  - Control regime figures (`results/figures/control`): Smith expert trials 431 to 1128 (mean 747), after the lift
    386 to 1196; the Smith criterion (lower 90 percent bound above chance) is more lenient than the sandbox proxy
    (moving performance above 0.8), so the same runs read 750 instead of 1140 trials. All panels reproduce; the gain
    panel (4f) shows the go / no-go neurons at gains 6 to 9 at texture time, the cue neuron at about 4.
  - Rule figures at the current best settings of the slow-policy condition (`hebb` 0.128 / 16 / theta_0 4 / lr_ap
    0.064; `bcm` 0.096 / tau 50 / e0 0.5 / lr_ap 0.032; `calcium` 0.001 / 0.5 / 1.5 / gamma_d 1.0 / beta 2 / lr_ap
    0.128): Smith expert means 651 (hebb, 2 of 10 seeds diverge at trials 321 and 614), 664 (bcm, none diverge),
    895 (calcium, 2 diverge at about 1200) versus 747 for the control. Selectivity picture: the control keeps a
    graded code (late response selectivity proportional to the initial basal selectivity, 6 percent go and 6
    percent no-go selective neurons); `hebb` prunes to a sparse code of about 2 go and 2 no-go detectors per seed
    (basal selectivity exactly +-1, responses at +-9 = the maximal gain) and silences 98.5 percent of the neurons;
    `bcm` similar (5 texture detectors, 130 distractor detectors per seed, 93 percent unresponsive); `calcium` is
    intermediate (graded but sharpened, 9 texture and 100 distractor detectors, 90 percent unresponsive).
- Step 2 at the chosen base condition (policy 0.016, TD 0.056, trace 0.256; sweeps `s3_rules`, `s3_inh`,
  `s3_refine`, `s3_fin10`, `s3_fin_inh`, `s3_stab`, `s3_stab_inh`, summaries only; control 1138 / 1140 by the
  proxy): the rules are less stable here than at the fast policy rate, because the policy takes long to learn and
  the sharpening has time to run away. 10-seed finalists (no-inhibition / after lift, runs lost of 20): `hebb`
  0.128 / 16 / 4 / lr_ap 0.064: 747 / 836, 9 lost (2 divergences and slow seeds without inhibition, 1 divergence
  and over-pruned seeds with); `bcm` 0.096 / 50 / 0.5 / 0.032: 919 / 1006, 5 lost (post-lift collapse in 4 of 10);
  `calcium` 0.001 / 0.5 / 1.5 / 1.0 / 0.128: 883 / 908, 8 lost (2 divergences in each protocol). Stability round
  (lower `lr_ap` 0.032 to 0.064, `ca_theta_p` 2.0, `theta_bas_0` up to 18, `bcm_e0_scale` 0.25, 5 seeds each
  protocol): the only settings without any lost seed barely act (`calcium` 0.0005 / 0.5 / 2.0 / 1.0 / 0.064:
  1011 / 1100 vs control 1138 / 1140; `bcm` at `lr_ap` 0.016: 1284 / 1545). `hebb` at this condition is
  asymmetric in the other direction (352 to 558 after the lift vs 871 to 1157 without: 1800 blind trials of
  pruning leave a sharpened network that then learns twice as fast) and always loses 2 to 4 of 10 runs. So at
  policy 0.016 no rule gives a large, clean, symmetric speed-up; the user has to decide between a modest clean one
  (calcium with a high LTP threshold, about 10 percent), accepting 10 to 20 percent lost seeds, or a faster base
  condition.
- Base condition rebuilt from structure (user, 2026-09-30): a slow policy rate is wanted; the control must be
  robust (bottom-up plasticity must never diverge an agent); basal weights are hard-constrained to [0, 1] (every
  sandbox bound now clamps to [0, 1], and the normalizing bounds `l1`, `pre`, `both` preserve the initial totals);
  the initialization was chosen on structural grounds before any learning: lognormal synaptic strengths
  (`init` lognormal, `init_sigma` 2), per-neuron normalized (`init_norm` neuron) and scaled to a total input of
  1.5 per neuron (`init_scale` 1.5), clipped at 1 (`init_clip` 1), with 20 distractors, which gives one third
  texture-responsive neurons at initialization by the manuscript's criterion (mean texture-time response above 0.1;
  responsiveness is set by the mean weight, so with the paper's convention of total input 1 the one-third point is
  at 10 distractors, and 20 to 30 distractors need a total of 1.5 to 1.9; unnormalized lognormals have no scale and
  a tail of single synapses of weight 10 to 50, per-stimulus normalization bounds totals but not synapses, per-neuron
  normalization tames the tail). Stability test: `w_bas_scale` multiplies the initial weights after the thresholds
  are set; the value estimator's effective step grows with the square of the drive, so a control's margin in drive
  amplitude is about the square root of the ratio between the divergence TD rate and its own. Old-init evidence
  (`c4_stab`, `c5_scale`, `c6_init`): at TD 0.016 or more every control diverges at twice the drive; at 0.008 it
  survives twice but not four times; the rules with the old init doubled the squared drive per texture (calcium),
  i.e. a 1.4-fold amplitude.
  - At the new init (sweeps `c7_ctrl`, `c8_ctrl`, `c9_ctrl10`, `c9_ctrl_inh`, `c9_margin`; policy 0.008 or 0.016,
    trace 0.256): TD 0.004 gives 930 to 1080 trials, survives 1.41- and 1.7-fold drive in every seed (and learns 2
    to 3 times faster there), diverges at 2-fold; TD 0.003 gives 1180 to 1400, loses 2 to 3 of 5 at 2-fold; TD
    0.002 survives 2-fold but needs 1600 to 1760 trials. Trace 1.0 and apical 0.064 to 0.256 change little.
  - Recommended base condition: policy 0.016, TD 0.0035, apical 0.128, trace 0.256: 1004 trials (10 seeds) and
    1053 after the lift (5 seeds; ratio 1.05), performance 0.52 during inhibition, no divergence at 1.41- and
    1.7-fold drive; one run of 15 is a slow seed beyond the horizon, not a divergence. The margin therefore covers
    the drive growth the clamped rules showed; a 2-fold amplitude margin would cost the target (TD 0.002, 1650).
- Population set to 1000 neurons (user, 2026-09-30, to reduce the seed lottery) and the control retuned (sweeps
  `c10_nz1000`, `c11_*`, `c12_theta`, `c13_*`; summaries only). Structure at 1000 neurons: with the lognormal init
  (sigma 2, per-neuron normalized, clipped at 1) one third texture-responsive needs a total input of 1.89 at 30
  distractors (2.26 at 50, 2.58 at 100); up to 50 distractors every seed has 3 to 5 saturated texture synapses per
  texture and 19 to 31 texture-dominant neurons per texture, at 75 to 100 some seeds again lack a strong texture
  synapse. 50 distractors at its one-third scale learns too slowly (fails within 2500 trials at any TD rate) and
  needs a total of 3.0 to 3.8 (65 to 83 percent responsive) to learn, so the user chose 30 distractors. A run of
  2500 trials takes about 11 s at this size.
  - Final control (user-approved regime, rates tuned here): `n_z` 1000, `n_noise` 30, `init` lognormal,
    `init_sigma` 2, `init_norm` neuron, `init_scale` 1.89, `init_clip` 1, `theta_0` 4, `lr_policy` 0.016,
    `lr_td` 0.0008, `lr_ap` 0.128, `lr_trace` 0.256: 1038 trials (10 seeds, no losses), 1090 after the lift (5
    seeds, ratio 1.05, chance during inhibition), no divergence at 1.7-fold drive (486 trials there), divergence at
    2-fold. TD 0.001 gives 855 / 915 with the same margin, 0.0006 gives 1305 / 1354 and survives 2-fold. The trace
    rate (0.128 to 0.512) and the apical rate (0.128 vs 0.256) change nothing; `theta_0` 8 stops learning;
    `theta_0` 2 is 25 percent faster but has no margin at all (1.7-fold diverges in every seed) and a post-lift
    ratio of 1.2, so 4 stays. Policy 0.008 is 5 to 10 percent slower with the same margin.
- Rule tuning at the final control (1000 neurons, 30 distractors; sweeps `r5_rules`, `r5_ext`, `r5_top10`,
  `r5_top_inh`, `r5_top_long`, `r5_ext10`, `r5_ext_inh`, `r5_long2`; policy 0.016, TD 0.0008, trace 0.256,
  `theta_0` 4 fixed; `lr_ap`, `lr_bas`, constants and the bound free; all bounds clamp to [0, 1]): not a single
  divergence in 120 + 24 grid configurations, 10-seed confirmations, the inhibition protocol or 4000-trial runs.
  Control: 1038 / 1090 (no inhibition / after the lift). Finalists (10 seeds / 5 seeds after the lift, ratio):
  - `hebb`, soft bound, `lr_bas` 0.256, `theta_bas_0` 8, `lr_ap` 0.128: 473 / 464 (0.98), stable to 4000 trials
    (10 seeds); prunes the total input per neuron from 1.84 to 1.14 and forms 25 tone, 66 texture and 718
    distractor one-hot detectors of 1000 neurons. With the combined bound (`both`, `lr_bas` 0.064, `theta_bas_0`
    32) it is faster (356 / 338) but loses 1 of 10 seeds late in a 4000-trial run, so the soft bound is the one.
  - `bcm`, soft bound, `lr_bas` 0.256, `bcm_tau` 200, `bcm_e0_scale` 0.5, `lr_ap` 0.128: 427 / 414 (0.97), stable
    to 4000 trials (10 seeds; `lr_bas` 0.512 with `bcm_tau` 200 diverges in every seed, with `bcm_tau` 100 it is
    417 / 346). The earlier finalist `lr_bas` 0.128, `bcm_tau` 100: 515 / 440 (0.85).
  - `calcium`, `pre` bound (each stimulus keeps its total outgoing weight), `lr_bas` 0.004, `ca_ltd_frac` 0.5,
    `ca_theta_p` 1.5, `ca_gamma_d` 1.0, `ca_beta` 2, `lr_ap` 0.128: 639 / 431 (0.67); faster basal rates (0.008
    to 0.016) give 528 to 578 without inhibition but the same asymmetry (0.62 to 0.71), so calcium keeps learning
    faster after the lift than without inhibition at every setting (its sharpening runs at gain 1 during the
    inhibited trials). The `l1` and `both` bounds are slower for calcium; `pre` is best.
  - Figures: `results/figures/{control2,hebb2,bcm2,calcium2}` (17 panels each, PNG committed; rebuilt on
    2026-09-30 with the final code and finalists, `bcm2` now at `lr_bas` 0.256 / `bcm_tau` 200) with the Smith
    expert trials (mean +- sd, no inhibition / after the lift): control 952 +- 263 (702 to 1624) / 1002 +- 246,
    hebb 394 +- 65 / 232 +- 108, bcm 346 +- 69 / 337 +- 65, calcium 564 +- 81 / 357 +- 90, no divergence. By
    the Smith criterion hebb and calcium are faster after the lift than without inhibition (the sandbox proxy
    gave hebb a ratio of 0.98); bcm is symmetric. `compare_control2_hebb2_bcm2_calcium2.png` and
    `fs12cd_wide_control2_hebb2_bcm2_calcium2.png`; at the new setting bcm forms about 95 texture and 40 tone
    detectors and its distractor detectors overshoot to 850 before settling at 600.
    Selectivity: the control keeps 42 percent of the neurons texture-responsive (29 percent non-selective, 6 and 7
    percent go / no-go selective, responses proportional to the initial basal selectivity, up to +-9 with gain);
    hebb and bcm end with 86 to 89 percent unresponsive and about 3.5 percent go and 3.5 percent no-go detectors at
    the maximal gain (responses at +-9, basal selectivity exactly +-1), i.e. a sparse code; calcium keeps a graded
    code (64 percent unresponsive, 22 percent non-selective, 7 percent go, 7 percent no-go) with responses spread
    between 0 and +-9. At this population size the initial control already has 40 texture and 620 distractor
    detectors per 1000 neurons, and the rules mostly convert the graded middle into either detectors or silence.
- Current state (2026-10-01, read this first; the "Open decision" and "Todo list" bullets below are from
  September 25 to 29 and superseded): two control conditions at 1000 neurons, 30 distractors, lognormal init
  (`init_sigma` 2, per-neuron normalized, `init_scale` 1.89, `init_clip` 1), `lr_policy` 0.016, `lr_trace` 0.256.
  Slow (robust, 1.7-fold drive margin): `theta_0` 4, `lr_td` 0.0008, `lr_ap` 0.128 (control 1038 / 1090 trials by
  the proxy) with `hebb` soft 0.256 / threshold 8 (473 / 464), `bcm` soft 0.256 / tau 200 / e0 0.5 (427 / 414),
  `calcium` pre 0.004 / 0.5 / 1.5 / 1.0 / beta 2 (639 / 431), all `lr_ap` 0.128; figure sets `*2`. Fast (at the
  divergence edge, no margin): `theta_0` 3, `lr_td` 0.0035, `lr_ap` 0.512 (control 306 / 372) with `hebb` soft
  0.512 / threshold 10 (208 / 178), `bcm` soft 0.256 / tau 50 / e0 0.25 (220 / 197), `calcium` l1 0.001 / 0.5 /
  2.0 / 2.0 / beta 2 (288 / 266), all `lr_ap` 0.512; figure sets `*3`. No divergence anywhere in either
  condition. The two-condition summary is `sandbox/results/figures/speedup_slow_fast.png`. The main code is
  untouched. Next: the user picks the rule(s) and condition for the figure, then port rule, init, population and
  rates into `helper.py` / `simulate_seed()`, regenerate `results_bup.pickle` and the performance traces, design
  the panels.
- Fastest control at the same environment (user, 2026-10-01: "how good can the control be in principle", to test
  whether the rules only pick up the inefficiency of a 1000-trial control; sweeps `o1_ctrl`, `o2_fine`, `o2_theta`,
  `o2_trace`, `o3_long`, `o3_inh`, summaries only; `sweep.py --summary-only` was added because 20-seed 4000-trial
  sweeps do not fit on the disk, a slim seed being about 65 MB): all rates and `theta_0` free, 1000 neurons, 30
  distractors, lognormal init unchanged. The policy rate cannot be raised at this init: 0.032 is slower than 0.016,
  0.064 loses seeds (final performance 0.79) and 0.128 or more never learns, so the slow policy is also the
  fastest. The TD rate sets the speed and the divergence edge: at `theta_0` 4, `lr_ap` 0.256 (a faster apical rate
  is more stable near the edge than 0.128), TD 0.005 learns in 375 trials (20 seeds over 4000 trials and 10 seeds in
  the inhibition protocol, no divergence; 438 after the lift), TD 0.006 in 354 but loses 1 run of 30, 0.007 or more
  diverges. `theta_0` 3 is faster (its edge is lower: TD 0.004 loses 2 to 8 of 30 runs, 0.005 diverges in 8 of
  10, `theta_0` 2 diverges everywhere): `theta_0` 3, `lr_ap` 0.512, TD 0.0035, policy 0.016, trace 0.256 learns in
  306 trials (20 seeds, 4000 trials, no divergence) and 372 after the lift (10 seeds, chance during inhibition),
  3.4 times faster than the 1000-trial control; with `lr_ap` 0.256 the same rates lose 2 of 30 runs. The trace
  rate (0.128 to 1.0) changes nothing. Chosen fast control: `theta_0` 3, `lr_ap` 0.512, `lr_td` 0.0035,
  `lr_policy` 0.016, `lr_trace` 0.256 (figure set `results/figures/control3`, Smith expert 230 +- 56 trials,
  267 after the lift). `figures.py` now skips the Tukey test of the phase boxplots when a learning phase has
  fewer than two seeds (fast learners have an empty learning phase), which made two panels fail.
- Rules at the fast control (2026-10-01; protocol as before: policy 0.016, TD 0.0035, trace 0.256, `theta_0` 3
  fixed, `lr_ap`, `lr_bas`, bound and constants free; sweeps `fc1_*` (5 seeds), `fc2_hebb`, `fc2_rescue`, `fc2_ctrl4`,
  `fc3_*` (10 seeds), `fc4_long` and `fc4_inh` (10 seeds, 4000 trials, confirmation), summaries only; the `fc` prefix
  avoids the September `f*` sweep names of the 203-neuron pass). The slow
  condition's finalists all diverge here: `bcm` (`bcm_e0_scale` 0.5) and `calcium` (`pre` bound, `ca_theta_p` 1.5,
  `ca_gamma_d` 1.0) in every seed at every `lr_bas` from 0.001 (calcium) or 0.064 (bcm) upwards, `hebb` with
  `theta_bas_0` 6 or 8 in every seed; the control sits at its TD divergence edge, so any rule that raises the drive
  pushes it over. All three rules can be retuned so that they do not: more pruning for `hebb` (`theta_bas_0` 10 to
  12; 9 diverges), a higher sliding threshold for `bcm` (`bcm_e0_scale` 0.25, `bcm_tau` 50; `bcm_tau` 100 or more
  with 0.25 loses 1 to 10 of 10 seeds, 0.125 is stable but slower at 253 to 422), and for `calcium` the `l1` bound
  with a higher LTP threshold and stronger depression (`ca_theta_p` 2, `ca_gamma_d` 2; `pre` and `soft` are at the
  control's speed or diverge). Confirmed finalists (10 seeds, 4000 trials / 10 seeds after the lift, no divergence
  in any run; control 307 / 372, chance during inhibition):
  - `hebb` soft, `lr_bas` 0.512, `theta_bas_0` 10, `lr_ap` 0.512: 208 / 178 (ratio 0.85; 32 percent faster than
    the control; `theta_bas_0` 11: 227 / 200). Prunes the total input per neuron to 0.77 and forms 14 tone, 49
    texture and 578 distractor detectors.
  - `bcm` soft, `lr_bas` 0.256, `bcm_tau` 50, `bcm_e0_scale` 0.25, `lr_ap` 0.512: 220 / 197 (0.90; 28 percent
    faster; `lr_bas` 0.128: 245 / 206). Total input 0.97, 12 tone, 62 texture, 782 distractor detectors.
  - `calcium` l1, `lr_bas` 0.001, `ca_ltd_frac` 0.5, `ca_theta_p` 2.0, `ca_gamma_d` 2.0, `ca_beta` 2, `lr_ap`
    0.512: 288 / 266 (0.92; 6 percent faster); `lr_bas` 0.002 gives 274 / 352 with final performance 0.985 in the
    inhibition protocol (5 texture detectors left after the blind trials), so 0.001 is the figure setting. It forms
    no tone detectors and 28 texture detectors.
  - Reading: at a control that is as fast as it can be without diverging, the pruning rules still cut the learning
    time by about 30 percent (versus 60 to 65 percent at the 1000-trial control), calcium by 6 percent (versus 40
    percent), and all of them need constants that keep the population drive below the control's: the rules' gain
    is bounded by the TD estimator's stability, not by the control's inefficiency alone. At the second-fastest clean
    control (`theta_0` 4, `lr_ap` 0.256, TD 0.005; 354 trials; sweep `fc2_ctrl4`) `hebb` at `theta_bas_0` 12 gives
    273 and `bcm` at `bcm_e0_scale` 0.25 303, while the slow-condition finalists diverge there too.
  - Figure sets `results/figures/{control3,hebb3,bcm3,calcium3}` (Smith expert trials, no inhibition / after the
    lift, mean +- sd over 10 seeds, no divergence: control 230 +- 56 / 267 +- 85, hebb 136 +- 23 / 85 +- 24, bcm
    148 +- 26 / 104 +- 22, calcium 213 +- 42 / 168 +- 38; the slow condition for comparison: control 952 / 1002,
    hebb 394 / 232, bcm 346 / 337, calcium 564 / 357), comparison `compare_control3_hebb3_bcm3_calcium3` and
    `fs12cd_wide_control3_...` (hebb and bcm again end with 92 to 94 percent unresponsive neurons and 2.5 to 3.5
    percent go and no-go detectors, calcium with 75 percent unresponsive and a graded 7 percent go / 7 percent
    no-go), and the two-condition summary `speedup_slow_fast.png` (`sandbox/speedup.py`: mean Smith performance
    traces without inhibition and in the inhibition protocol, per-seed Smith expert trials without inhibition and
    after the lift, for `control2`, `hebb2`, `bcm2`, `calcium2` versus the `*3` sets). By the Smith criterion the
    speed-up at the fast control is 41 percent (hebb), 36 percent (bcm) and 7 percent (calcium) without inhibition
    and 60 to 68 percent (hebb, bcm) and 37 percent (calcium) after the lift, versus 59 / 64 / 41 percent and 77 /
    66 / 64 percent at the slow control. Every rule is faster after the lift than without inhibition at the fast
    control (ratios 0.62 to 0.79; the control's is 1.16).
- Open decision: the shared environment can be 203 neurons at `init_power` 12 (100 neurons also works for calcium but
  not for any gated rule). Calcium, `hebb` with `theta_bas_0` 16 and constant LTD with L1 (`rule` none, `lr_dep`
  0.016 `const`) are the settings that beat the control by a wide margin (pruning 392 to 547 trials, calcium 328
  to 534, `hebb` 553 to 722, control 1033 to 1235); only `hebb` passes both inhibition criteria without post-lift
  collapse. `us` gives a modest, clean improvement; `gated_hebb` and `burst` are
  indistinguishable from tuning the learning rates. Still open: accept sharpening or pruning that also runs at gain 1, or add
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
