# Covasim

COVID-19 agent-based simulator. A stochastic individual-based model that can be used for Covid-19 (novel coronavrius, SARS-CoV-2) epidemic projections, scenario interventions, etc., and adapted to different contexts (e.g. the Diamond Princess cruise ship, cities, countries).


## Requirements

Python >=3.6 (64 bit).


## Installation

Standard Python package installation: `python setup.py develop`.

The module should then be importable as `import covasim`.

If you want to run as just a standalone model (not within a web application), you can instead run `python setup.py develop nowebapp`. To install optional dependencies, you can also do `python setup.py develop full`, although note that this will likely fail since it relies on private packages.


## Usage

Simplest usage is `python examples/run_sim.py`. This will create a figure. See also `covasim/covasim/base/run_scenarios.py` for a more complete usage example.


## Structure

The package contains multiple different flavors of Covasim. A flavor can have its own parameters file (`parameters.py`) and/or slight variations to the model (`model.py`). Different versions are:
* `base` -- **this is the main version of the code**, and is what's available in `import covasim`
* `cruise_ship` -- model specific for the Diamond Princess cruise ship
* `framework` -- classes and functions defining the structure of the module, but not to be used directly
* `cova_webapp` -- for running an interactive webapp