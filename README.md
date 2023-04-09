# Deep Q-Learning with Keras and Gym

This project is for the second assignment of the course Reinforcement Learning at Leiden University.
The goal of this assignment is to implement a Deep Q-Learning algorithm to solve the CartPole problem from OpenAI Gym.

## Contributors

- Jasper Steenbergen
- Josef Hamelink
- Willem Venemans

## Setup

Note that for macOS users, the `tensorflow-macos` package is not available for Python 3.11.
Python 3.10 is recommended, also for other platforms.

Lower than 3.10 might cause issues with `tensorflow` and the recent improvements to type hinting, which are used in this project.

1. Clone the repository
2. (Recommended) Create a virtual environment with Python 3.10
  - e.g., `python3.10 -m venv venv`
3. Activate the virtual environment
  - e.g., `source venv/bin/activate`
  - when working with an inferior shell such as powershell, use `venv\Scripts\activate`, you might even need to use the `.ps1` or `.bat` script
4. Install the requirements
  - `pip install -e .` (should work on all platforms)

## Script usage

From the root directory, you can now run the package with the command `dql`.

When using for the first time, run with the help flag to see the available arguments: `dql -h` or `dql --help`.

For light use (single run or handful of runs with low episode count), an example command would be:

```bash
dql -nr 5 -ne 5000 -es boltzmann -V -R
```

This will run the algorithm 5 independent times for 5000 episodes each, using the Boltzmann exploration strategy, with verbose output and the trained agent's behaviour being rendered after each repetition.

For more intensive experiments, we might run into memory issues.
To get around this, you should run the `dqlw.sh` script, which will call `dql` with the arguments you provide for `-nr {x}` independent times, i.e. it calls `dql` `x` times with the same arguments.

For example:

```bash
cd shellscripts
./dqlw.sh -nr 10 -ne 25000 -es boltzmann -I boltzmann
# you probably do not want verbose output and rendering here
```

In _shellscripts_, you will find all of the scripts that can be used to replicate the experiments.
It should be noted that you might get different results than the ones in the report, because fixing a random seed for the environment is not possible (to our knowledge).

## Results

Results are automatically saved under _data/yymmdd-hhmmss/_ unless you specify a run id with `-I {id}`.

Example structure of results directory:

```bash
data
└── boltzmann
    ├── summary.json
    ├── actions.npz
    ├── losses.npz
    ├── rewards.npz
    └── behaviour_models
        ├── 1.h5
        ├── 2.h5
        ...
        └── 10.h5
```

If the `-TN` (target network) flag was set, the target_models will also be saved, just like the behaviour_models.

## Notebooks

The notebooks are used for data analysis and visualisation.
They work by importing a `DataManager` class from the `dql.utils` module which allows them to easily load the data from the data directory.
It is important to note that the data directory is not included in the repository, so you will have to run the scripts first to generate the data.
Maybe in the future we will host the data somewhere so it can be pulled, but for now you will have to generate it yourself.
