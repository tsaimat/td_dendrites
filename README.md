# Error representations in apical dendrites of neocortical layer 5 neurons during learning

## Description

This program runs the simulations and plots the corresponding panels described in the manuscript '[Error representations in apical dendrites of neocortical layer 5 neurons during learning](https://www.biorxiv.org/content/10.1101/2021.12.28.474360v3)'. This means simulating models of mice with L5 sensory pyramidal neurons learning a go/no-go sensory discrimination task from a sensory representation plagued by distractor signals.

### Installation

These are instructions to setup a functioning conda environment. If conda is not installed yet, do so following these [instructions](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) (linux recommended). Then make sure conda is up-to-date.
```
conda update conda --all
```

Create a conda environment with your favorite name, here `myenv`
```
conda create -n myenv python=3.10.8
```

Activate the conda environment 
```
conda activate myenv
```

Install the required libraries
```
pip install -r requirements.txt
```

Matlab is not needed: the performance traces are extracted with a Python port of the method of Smith et al. (2004). The original Matlab scripts are kept in the folder `Smith` (tested with R2022b) and can be used instead.

### Execution

Before rerunning the simulations, you can test whether the installation was successful. Just enter
```
python main.py plot
```

This will plot all the panels (to the folder `panels`) with the existing simulation results. This is possible as the `main` branch contains pre-simulated results in the `results` folder, which are ~7GB and stored with [Git LFS](https://git-lfs.com) (install it before cloning). If you want to avoid downloading these files and simulate them yourself, you can clone the `clean` branch instead of the `main` branch. To rerun all the simulations, enter
```
python main.py simulate
```

This will overwrite any simulation outcomes present in the `results` folder (about an hour on an 8-core machine, most of it for the extraction of the performance traces, which runs in parallel on all available cores). The simulations end by saving the trial outcomes to `Smith/outcomes_*.mat`, extracting the performance traces with the Python port of the code from Smith et al. (2004) into `Smith/performances.mat` and integrating them into the pickle files in the `results` folder, after which all of the panels can be replotted. To extract the performance traces again, enter
```
python main.py perf
```

To use the original Matlab code instead, run `Smith/getperfs.m` with Matlab after the simulations (it reads the same `Smith/outcomes_*.mat` and overwrites `Smith/performances.mat`) and then enter
```
python main.py perf matlab
```

If any step is interrupted or skipped, the results files can become corrupted and can only be recovered by rerunning the simulation from the start or downloading the uncorrupted results from the github repository.
