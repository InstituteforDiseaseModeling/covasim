=======
Covasim
=======

Covasim is a stochastic agent-based simulator designed to be used for COVID-19
(novel coronavirus, SARS-CoV-2) epidemic analyses. These include projections of
indicators such as numbers of infections and peak hospital demand. Covasim can
also be used to explore the potential impact of different interventions, including
social distancing, school closures, testing, contact tracing, and quarantine.

The Covasim webapp is available at https://app.covasim.org.

Questions or comments can be directed to covasim@idmod.org, or on this project's
GitHub_ page. Full information about Covasim is provided in the documentation_.

.. _GitHub: https://github.com/institutefordiseasemodeling/covasim
.. _documentation: https://docs.covasim.org

.. contents:: Contents
   :local:
   :depth: 2


Requirements
============

Python >=3.6 (64-bit). (Note: Python 2 is not supported.)

We also recommend, but do not require, using Python virtual environments. For
more information, see documentation for venv_ or Anaconda_.

.. _venv: https://docs.python.org/3/tutorial/venv.html
.. _Anaconda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


Quick start guide
==================

Install with ``pip install covasim``. If everything is working, the following Python commands should bring up a plot::

  import covasim as cv
  sim = cv.Sim()
  sim.run()
  sim.plot()


Detailed installation instructions
==================================

1.  Clone a copy of the repository. If you intend to make changes to the code,
    we recommend that you fork it first.

2.  (Optional) Create and activate a virtual environment.

3.  Navigate to the root of the repository and install the Covasim Python package
    using one of the following options:

    *   To install with web app support (recommended)::

          python setup.py develop

    *   To install as a standalone Python model without webapp support::

          python setup.py develop nowebapp

    *   To install Covasim and optional dependencies (be aware this may fail
        since it relies on private packages), enter::

          python setup.py develop full

    The module should then be importable via ``import covasim``.


Usage examples
==============

There are several examples in the `examples` directory. These can be run as
follows:

* ``python examples/simple.py``

  This example creates a figure using default parameter values.

* ``python examples/run_sim.py``

  This shows a slightly more detailed example, including creating an intervention and saving to disk.

* ``python examples/run_scenarios.py``

  This shows a more complex example, including running an intervention scenario, plotting uncertainty, and performing a health systems analysis.


Module structure
================

All core model code is located in the ``covasim`` subfolder; standard usage is
``import covasim as cv``. The other subfolders, ``cruise_ship``, ``data``, and ``webapp``, are
also described below.

The model consists of two core classes: the ``Person`` class (which contains
information on health state), and the ``Sim`` class (which contains methods for
running, calculating results, plotting, etc.).

The structure of the ``covasim`` folder is as follows:

* ``base.py``: The ``ParsObj`` class, plus basic methods of the ``BaseSim`` class, and associated functions.
* ``defaults.py``: The default colors, plots, etc. used by Covasim.
* ``interventions.py``: The ``Intervention`` class, for adding interventions and dynamically modifying parameters.
* ``misc.py``: Miscellaneous helper functions.
* ``parameters.py``: Functions for creating the parameters dictionary and loading the input data.
* ``population.py``: The ``People`` class, for handling updates of state for each person.
* ``plotting.py``: Plotly graphs to supplement the built-in Matplotlib graphs.
* ``population.py``: Functions for creating populations of people, including age, contacts, etc.
* ``requirements.py``: A simple module to check that imports succeeded, and turn off features if they didn't.
* ``run.py``: Functions for running simulations (e.g. parallel runs and the ``Scenarios`` class).
* ``sim.py``: The ``Sim`` class, which performs most of the heavy lifting: initializing the model, running, and plotting.
* ``utils.py``: Functions for choosing random numbers, many based on Numba, plus other helper functions.
* ``version.py``: Version, date, and license information.


Cruise ship
-----------

A version of the Covasim model specifically adapted for modeling the Diamond
Princess cruise ship. It uses its own parameters file (``parameters.py``) and has
slight variations to the model (``model.py``).


Data
----

This folder contains loading scripts for the epidemiological data in the root ``data`` folder, as well as data on age distributions for different countries and household sizes.



Webapp
------

For running the interactive web application. See the `webapp README`_ for more information.

.. _webapp README: https://github.com/InstituteforDiseaseModeling/covasim/tree/master/covasim/webapp


Other folders
=============

Please see the readme in each subfolder for more information.


bin
---

This folder contains a command-line interface (CLI) version of Covasim; example usage::

  covasim --pars "{pop_size:20000, pop_infected:1, n_days:360, rand_seed:1}"

Note: the CLI is currently not compatible with Windows. You will need to add
this folder to your path to run from other folders.


data
----

Scripts to automatically scrape data (including demographics and COVID epidemiology data),
and the data files themselves (which are not part of the repository).


docker
------

This folder contains the ``Dockerfile`` and other files that allow Covasim to be
run as a webapp via Docker.


examples
--------

This folder contains demonstrations of simple Covasim usage.


licenses
--------

Licensing information and legal notices.


tests
-----

Integration, development, and unit tests.


sweep
-----

Utilities for hyperparameter sweeps, using `Weights and Biases`_. See the `sweep README`_ for more information.

.. _Weights and Biases: https://www.wandb.com/
.. _sweep README: ./sweep


Disclaimer
==========

The code in this repository was developed by IDM to support our research in
disease transmission and managing epidemics. We’ve made it publicly available
under the Creative Commons Attribution-Noncommercial-ShareAlike 4.0 License to
provide others with a better understanding of our research and an opportunity to
build upon it for their own work. We make no representations that the code works
as intended or that we will provide support, address issues that are found, or
accept pull requests. You are welcome to create your own fork and modify the
code to suit your own modeling needs as contemplated under the Creative Commons
Attribution-Noncommercial-ShareAlike 4.0 License. See the contributing and code of conduct
READMEs for more information.