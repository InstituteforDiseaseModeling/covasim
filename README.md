# Covasim

Covasim is a stochastic agent-based simulator designed to be used for COVID-19
(novel coronavirus, SARS-CoV-2) epidemic analyses. These include projections of
indicators such as numbers of infections and peak hospital demand. Covasim can
also be used to explore the potential impact of different interventions.

1. [Requirements](#Requirements)
2. [Quick start guide](#Quickstart)
3. [Detailed installation instructions](#Installation)
4. [Detailed usage](#Usage)
5. [Structure](#Structure)
	5.1. [covasim](#covasim)
	5.2. [cruise_ship](#cruise_ship)
	5.3. [webapp](#webapp)
6. [Other folders](#Other)
	6.1. [bin](#bin)
	6.2. [docker](#docker)
	6.3. [examples](#examples)
	6.4. [licenses](#licenses)
	6.5. [tests](#tests)
	6.6. [sweep](#sweep)
7. [Disclaimer](#Disclaimer)


##  1. <a name="Requirements"></a>Requirements

Python >=3.6 (64-bit). (Note: Python 2 is not supported.)

We also recommend, but do not require, using Python virtual environments. For
more information, see documentation for [venv](https://docs.python.org/3/tutorial/venv.html) or [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).


##  2. <a name="Quickstart"></a>Quick start guide

Install with `pip install covasim`. If everything is working, the following Python commands should bring up a plot:

```python
import covasim as cv
sim = cv.Sim()
sim.run()
sim.plot()
```


##  3. <a name="Installation"></a>Detailed installation instructions

1.  Clone a copy of the repository. If you intend to make changes to the code,
    we recommend that you fork it first.

2.  (Optional) Create and activate a virtual environment.

3.  Navigate to the root of the repository and install the Covasim Python package
    using one of the following options:

    *   To install with webapp support (recommended):

        `python setup.py develop`

    *   To install as a standalone Python model without webapp support:

        `python setup.py develop nowebapp`

    *   To install Covasim and optional dependencies (be aware this may fail
        since it relies on private packages), enter:

        `python setup.py develop full`

    The module should then be importable via `import covasim`.


##  4. <a name="Usage"></a>Detailed usage

There are several examples in the `examples` directory. These can be run as
follows:

* `python examples/simple.py`

  This example creates a figure using default parameter values.

* `python examples/run_sim.py`

  This shows a slightly more detailed example, including creating an intervention and saving to disk.

* `python examples/run_scenarios.py`

  This shows a more complex example, including running an intervention scenario, plotting uncertainty, and performing a health systems analysis.


##  5. <a name="Structure"></a>Structure

All core model code is located in the `covasim` subfolder; standard usage is
`import covasim as cv`. The other subfolders, `cruise_ship` and `webapp`, are
also described below.

The model consists of two core classes: the `Person` class (which contains
information on health state), and the `Sim` class (which contains methods for
running, calculating results, plotting, etc.).


###  5.1. <a name="covasim"></a>covasim

The structure of the `covasim` folder is as follows:

* `base.py`: The `ParsObj` class, plus basic methods of the `BaseSim` class, and associated functions.
* `defaults.py`: The default colors, plots, etc. used by Covasim.
* `interventions.py`: The `Intervention` class, for adding interventions and dynamically modifying parameters.
* `parameters.py`: Functions for creating the parameters dictionary and loading the input data.
* `person.py`: The `Person` class.
* `population.py`: The `People` class, and functions for creating a population of people.
* `requirements.py`: A simple module to check that imports succeeded, and turn off features if they didn't.
* `run.py`: Functions for running simulations (e.g. parallel runs and the `Scenarios` class).
* `sim.py`: The `Sim` class, which performs most of the heavy lifting: initializing the model, running, and plotting.
* `utils.py`: Functions for choosing random numbers, many based on Numba, plus other helper functions.
* `version.py`: Version, date, and license information.


###  5.2. <a name="cruise_ship"></a>cruise_ship

A version of the Covasim model specifically adapted for modeling the Diamond
Princess cruise ship. It uses its own parameters file (`parameters.py`) and has
slight variations to the model (`model.py`).


###  5.3. <a name="webapp"></a>webapp

For running the interactive web application. See the [webapp readme](./covasim/webapp) for more information.


##  6. <a name="Otherfolders"></a>Other folders

Please see the readme in each subfolder for more information.

###  6.1. <a name="bin"></a>bin

This folder contains a command-line interface (CLI) version of Covasim; example usage:

```bash
covasim --pars "{pop_size:20000, pop_infected:1, n_days:360, rand_seed:1}"
```

Note: the CLI is currently not compatible with Windows. You will need to add
this folder to your path to run from other folders.

###  6.2. <a name="docker"></a>docker

This folder contains the `Dockerfile` and other files that allow Covasim to be
run as a webapp via Docker.

###  6.3. <a name="examples"></a>examples

This folder contains demonstrations of simple Covasim usage.

###  6.4. <a name="licenses"></a>licenses

Licensing information and legal notices.

###  6.5. <a name="tests"></a>tests

Integration, development, and unit tests.

###  6.6. <a name="sweep"></a>sweep

Utilites for hyperparameter sweeps, using [Weights and Biases](https://www.wandb.com/). See the [sweeps readme](./sweeps) for more information.


##  7. <a name="Disclaimer"></a>Disclaimer

The code in this repository was developed by IDM to support our research in
disease transmission and managing epidemics. We’ve made it publicly available
under the Creative Commons Attribution-Noncommercial-ShareAlike 4.0 License to
provide others with a better understanding of our research and an opportunity to
build upon it for their own work. We make no representations that the code works
as intended or that we will provide support, address issues that are found, or
accept pull requests. You are welcome to create your own fork and modify the
code to suit your own modeling needs as contemplated under the Creative Commons
Attribution-Noncommercial-ShareAlike 4.0 License.
