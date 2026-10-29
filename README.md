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

The simulation results (~8GB) are not stored in this repository. To generate them in the `results` folder, enter
```
python main.py simulate
```

This will overwrite any simulation outcomes present in the `results` folder. Each simulation runs its seeds in parallel on all available cores, extracts their performance traces with a Python port of the method of Smith et al. (2004) and saves the results once all its seeds are done (all simulations take about 90 minutes on an 8-core machine). If the simulations are interrupted, the results saved before stay complete; rerun `python main.py simulate` to regenerate the others. To extract the performance traces of the saved results again, enter
```
python main.py perf
```

Once the results exist, plot all the panels (to the folder `panels`, and those of the fast-learning regime to `panels/fast`) with
```
python main.py plot
```
