# COVASim

COVID-19 Agent-based Simulator. A stochastic individual-based model that can be used for Covid-19 (novel coronavrius, SARS-nCoV-2) epidemic projections, scenario interventions, etc., and adapted to different contexts (e.g. the Diamond Princess cruise ship, cities, countries).


## Requirements

Python >=3.6 (64 bit).


## Installation

Standard Python package installation: `python setup.py develop`.

The module should then be importable as `import covasim`.

If you want to run as just a standalone model (not within a web application), you can instead run `python setup.py develop nowebapp`.


## Usage

Simplest usage is `python scripts/run_sim.py`. This will create a figure. Different flavors are described below.


## Flavors

The package contains multiple different flavors of COVASim. Each flavor has its own parameters file (`parameters.py`) and slight variations to the model (`model.py`). Different versions are:
* `cova_base` -- not to be used directly, the base classes for the simulation
* `cova_cdc` -- for intervention scenarios for the CDC, uses detailed age mixing patterns for Seattle
* `cova_generic` -- **the latest version of the code**, not specific to a particular context
* `cova_cruise` -- for the Diamond Princess cruise ship, includes data on diagnoses and passengers and crew
* `cova_oregon` -- estimtaes for the Oregon Health Authority
* `cova_seattle` -- estimates for the Washington Department of Health
* `cova_webapp` -- for running an interactive webapp

More information is available in the `README.md` in each folder (coming soon).


### Oh god, why is there so much duplication?

Because if we get a new request from the CDC, we don't want to break Seattle. Consistency and flexibility is more important than textbook-perfect software architecture here. We'll consider combining later.


### I want to know more

Totally reasonable request. We are still working on the documentation.