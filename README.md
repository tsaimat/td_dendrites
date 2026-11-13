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

The performance traces are extracted with a Python port of the method of Smith et al. (2004), so no Matlab is needed.

### Execution

The simulation results (~10GB) are not stored in this repository. To run all the simulations and plot all the panels, enter
```
python main.py
```

This runs every simulation, saves its results in the `results` folder (overwriting any simulation outcomes present there) and then plots all the panels to the folder `panels` (those of the fast-learning regime to `panels/fast`). Each simulation runs its seeds in parallel on all available cores, extracts their performance traces with a Python port of the method of Smith et al. (2004) and saves the results once all its seeds are done (all simulations take about 90 minutes on an 8-core machine, the plotting a few minutes). If the simulations are interrupted, the results saved before stay complete.

Each step can also be run on its own:
```
python main.py simulate   # run all the simulations, including the extraction of the performance traces
python main.py perf       # extract the performance traces of the saved results again
python main.py plot       # plot all the panels from the saved results
```
