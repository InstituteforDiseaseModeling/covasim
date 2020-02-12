# covid_abm
A model for estimating Covid-19 (novel coronavrius, formerly nCoV-2019) parameters from cruise ship infection data.

## Requirements

Python >=3.7 (64 bit).


## Installation

Standard Python package installation: `python setup.py develop`.

The module should then be importable as `import covid_abm`.


## Usage

Usage is simply `python -i scripts/run_sim.py`. This will create a figure (and, by default, save it to disk). You can modify parameter values, including the random seed, in `covid_abm/parameters.py`. There are various tests in the `tests` folder.
