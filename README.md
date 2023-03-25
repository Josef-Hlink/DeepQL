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

## Usage

From the root directory, you can now run the package with the command `dql`.

When using for the first time, run with the help flag to see the available arguments: `dql -h` or `dql --help`.

If _setup.py_ did not work (step 4), please manually identify & install the dependencies and run the project with `python3.10 dql/main.py`.

Note that you will have to remove the dots from the imports in _dql/main.py_: `from .utils.module import func` $\to$ `from utils.module import func`.
