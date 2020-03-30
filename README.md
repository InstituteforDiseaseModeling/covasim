# Covasim

Covasim is a stochastic agent-based simulator designed to be used for COVID-19
(novel coronavirus, SARS-CoV-2) epidemic analyses. These include projections of
indicators such as numbers of infections and peak hospital demand. Covasim can
also be used to explore the potential impact of different interventions.


## Requirements

Python >=3.6 (64-bit). (Note: Python 2 is not supported.)

We also recommend, but do not require, using Python virtual environments. For
more information, see documentation for [venv](https://docs.python.org/3/tutorial/venv.html) or [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## Quick start guide

Install with `pip install covasim`. If everything is working, the following Python commands should bring up a plot:

```python
import covasim as cv
sim = cv.Sim()
sim.run()
sim.plot()
```


## Detailed installation instructions

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


## Detailed usage

There are several examples in the `examples` directory. These can be run as
follows:

* `python examples/simple.py`

  This example creates a figure using default parameter values.

* `python examples/run_sim.py`

  This shows a slighly more detailed example, including creating an intervention and saving to disk.

* `python examples/run_scenarios.py`

  This shows a more complex example, including running an intervention scenario, plotting uncertainty, and performing a health systems analysis.


## Structure

All core model code is located in the `covasim` subfolder; standard usage is
`import covasim as cv`. The other subfolders, `cruise_ship` and `webapp`, are
also described below.

The model consists of two core classes: the `Person` class (which contains
information on health state), and the `Sim` class (which contains methods for
running, calculating results, plotting, etc.).


### covasim

The structure of the `covasim` folder is as follows:

* `base.py`: The `ParsObj` class, plus basic methods of the `BaseSim` class, and associated functions.
* `healthsystem.py`: The `HealthSystem` class, for determining hospital capacity and treatment rates.
* `interventions.py`: The `Intervention` class, for adding interventions and dynamically modifying parameters.
* `parameters.py`: Functions for creating the parameters dictionary and loading the input data.
* `people.py`: The `Person` class, and functions to create a population of people.
* `requirements.py`: A simple module to check that imports succeeded, and turn off features if they didn't.
* `run.py`: Functions for running simulations (e.g. parallel runs and scenarios).
* `sim.py`: The `Sim` class, which performs most of the heavy lifting: initializing the model, running, and plotting.
* `utils.py`: Functions for choosing random numbers, many based on Numba, plus other helper functions.
* `version.py`: Version, date, and license information.


### cruise_ship

A version of the Covasim model specifically adapted for modeling the Diamond
Princess cruise ship. It uses its own parameters file (`parameters.py`) and has
slight variations to the model (`model.py`).


### webapp

For running the interactive web application: please see the `README.md` in that
folder for more information, including information on running locally and Docker
deployment.


## Disclaimer

The code in this repository was developed by IDM to support our research in
disease transmission and managing epidemics. We’ve made it publicly available
under the Creative Commons Attribution-Noncommercial-ShareAlike 4.0 License to
provide others with a better understanding of our research and an opportunity to
build upon it for their own work. We make no representations that the code works
as intended or that we will provide support, address issues that are found, or
accept pull requests. You are welcome to create your own fork and modify the
code to suit your own modeling needs as contemplated under the Creative Commons
Attribution-Noncommercial-ShareAlike 4.0 License.
