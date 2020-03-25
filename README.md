# Covasim

COVID-19 agent-based simulator. A stochastic individual-based model that can be used for Covid-19 (novel coronavrius, SARS-CoV-2) epidemic projections, scenario interventions, etc., and adapted to different contexts (e.g. the Diamond Princess cruise ship, cities, countries).


## Requirements

Python >=3.6 (64 bit).


## Installation

Standard Python package installation: `python setup.py develop`.

The module should then be importable as `import covasim`.

If you want to run as just a standalone model (not within a web application), you can instead run `python setup.py develop nowebapp`. To install optional dependencies, you can also do `python setup.py develop full`, although note that this will likely fail since it relies on private packages.


## Usage

Simplest usage is `python examples/simple.py`. This will create a figure. See also `examples/run_scenarios.py` for a more complete usage example.


## Structure

All core model code is located in the `covasim` subfolder; standard usage is `import covasim as cv`.

### covasim

The model consists of two core classes: the `Person` class (which contains information on health state), and the `Sim` class (which contains methods for running, calculating results, plotting, etc.).

The structure of the `covasim` folder, in the order imported, is as follows:

* `version.py`: Version and version date information.
* `requirements.py`: Check that imports succeeded, and turn off features if they didn't.
* `poisson_stats.py`: For comparing counts (e.g., actual number of positive diagnoses vs. predicted number).
* `utils.py`: Numeric utilities, mostly based on Numba, for choosing random numbers (plus other helper functions).
* `base.py`: The `ParsObj` class, plus basic methods of the `BaseSim` class, and associated functions.
* `parameters.py`: Functions for creating the parameters dictionary and populating correct attributes for people.
* `model.py`: The core class defining the model, namely `Person` and `Sim`. `Sim` inherits from `BaseSim` which inherits from `ParsObj` which inherits from `prettyobj`.

The package contains multiple different flavors of Covasim. A flavor can have its own parameters file (`parameters.py`) and/or slight variations to the model (`model.py`). 

The `README.md` in that folder contains more information on the parameters.

### cruise_ship

A version of the Covasim code specifically adapted for modeling the Diamond Princess cruise ship.

### webapp

For running the interactive webapp; please see the `README.md` in that folder for more information.