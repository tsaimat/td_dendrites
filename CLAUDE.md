# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Simulation code and figure generation for the manuscript "Error representations in apical dendrites of neocortical layer 5 neurons during learning" (Schoenfeld et al., bioRxiv 10.1101/2021.12.28.474360). A mouse agent learns a go/no-go texture discrimination task; L5 pyramidal neurons receive basal sensory input (task stimuli plus many distractors) and a top-down TD-error signal onto their apical dendrites, which multiplicatively gates somatic output. There is no test suite; correctness is checked by regenerating the panels in `panels/` and comparing them to the manuscript figures.

## Environment and commands

Python 3.10, dependencies pinned in `requirements.txt` (torch 1.13, numpy 1.23, scipy 1.9, matplotlib 3.7, statsmodels). Install with `pip install -r requirements.txt` inside a conda env. Matlab (tested with R2022b) is only needed to regenerate performance traces.

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

## Work in progress (as of 2026-09-24)

The `Simulation.BU_PLASTICITY` variant and the exploration sandbox are committed and pushed to the fork
(`tsaimat/td_dendrites`, commits `6f12b3c`, `ebb3d6e`, `09afe36`). State of the work:

- Main code is done: `N_Z_BU_PLASTICITY = 30` distractors, random L1-normalized `w_bas` init shared with the mixed model, soft-bounded basal plasticity `(1 - w) * w` in the trial loop, gain recording via a neuron-space mask, Matlab script preallocates from the file list. The default model reproduces the committed `results_default.pickle` exactly, and the mixed-model initialization is bit-identical to before.
- Data is stale: `results/results_bup.pickle`, `Smith/outcomes_bup.mat` and the `perf_bup` rows in `Smith/performances.mat` were generated when the variant was a no-op (identity init, so identical to default). Regenerate once a rule is chosen: `simulate_seeds(Simulation.BU_PLASTICITY)`, `save_trial_outcomes_matlab()`, `Smith/getperfs.m` in Matlab, then `python main.py perf`.
- The default, apical-inhibition and mixed results committed in `6f12b3c` are the same simulations as before; only the Matlab-derived performance traces and expert/learning trials shifted by re-running the estimator (typically 1 to 4 trials, one 93-trial outlier for mixed seed 1 where the confidence band hovers at chance).
- No panel plots the bottom-up results yet. `f4f_sen_dendrite` assumes neuron i is wired to stimulus i, so it is only meaningful for the default model.
- `sandbox/` is a self-contained, deletable package for trying out bottom-up plasticity rules and hyperparameter sweeps without changing `l5apical` (see `sandbox/README.md`). Its `simulate.py` mirrors the main trial loop; run `python -m sandbox.check_reproduces` after editing either (passes as of this commit). Sweep output goes to the gitignored `sandbox/results/`.
- Sandbox findings so far (300-trial smoke test only): `hard_clamped` and `oja` diverge to NaN at the default learning rate because total basal drive per neuron is unbounded; `soft_bounded`, `l1_renormalized`, `bcm` and `apical_gated` run stably. `l1_renormalized` moves the weights most and raises the task-stimulus weight fraction. No full-length sweep has been run yet.
- Suggested next steps: run the full-length rule sweep and a learning-rate sweep from the README on a multi-core machine, pick a rule, port it into `simulate_seed()` in `l5apical/simulations.py`, regenerate the bottom-up data, then delete `sandbox/`.
- `play.py` is a scratch script and `Overview.pdf` is a manuscript overview; both are committed but were not part of the original repo.
