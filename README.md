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

Before rerunning the simulations, you can test whether the installation was successful. Just enter
```
python main.py plot
```

This will plot all the panels (to the folder `panels`) with the existing simulation results. This is possible as the `main` branch contains pre-simulated results in the `results` folder, which are ~7GB and stored with [Git LFS](https://git-lfs.com) (install it before cloning). If you want to avoid downloading these files and simulate them yourself, you can clone the `clean` branch instead of the `main` branch. To rerun all the simulations, enter
```
python main.py simulate
```

This will overwrite any simulation outcomes present in the `results` folder. Each simulation runs its seeds in parallel on all available cores, extracts their performance traces with a Python port of the method of Smith et al. (2004) and saves the results once all its seeds are done (all simulations take about 35 minutes on an 8-core machine). To extract the performance traces of the saved results again, enter
```
python main.py perf
```

If the simulations are interrupted, the results saved before stay complete; rerun `python main.py simulate` to regenerate the others.
