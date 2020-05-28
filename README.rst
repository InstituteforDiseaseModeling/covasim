=======
Covasim
=======

Covasim is a stochastic agent-based simulator designed to be used for COVID-19
(novel coronavirus, SARS-CoV-2) epidemic analyses. These include projections of
indicators such as numbers of infections and peak hospital demand. Covasim can
also be used to explore the potential impact of different interventions, including
social distancing, school closures, testing, contact tracing, and quarantine.

The scientific paper describing Covasim is available at http://paper.covasim.org.
The recommended citation is:

    Kerr CC, Stuart RM, Mistry D, Abeysuriya RG, Hart G, Rosenfeld R, Selvaraj P, Núñez RC, Hagedorn B, George L, Izzo A, Palmer A, Delport D, Bennette C, Wagner B, Chang S, Cohen JA, Panovska-Griffiths J, Jastrzębski M, Oron AP, Wenger E, Famulare M, Klein DJ (2020). **Covasim: an agent-based model of COVID-19 dynamics and interventions**. *medRxiv* 2020.05.10.20097469; doi: https://doi.org/10.1101/2020.05.10.20097469.

The Covasim webapp is available at http://app.covasim.org.

Questions or comments can be directed to covasim@idmod.org, or on this project's
GitHub_ page. Full information about Covasim is provided in the documentation_.

.. _GitHub: https://github.com/institutefordiseasemodeling/covasim
.. _documentation: https://docs.covasim.org


.. contents:: **Contents**
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
``import covasim as cv``. The other subfolders, ``data``, and ``webapp``, are
also described below.

The model consists of two core classes: the ``Person`` class (which contains
information on health state), and the ``Sim`` class (which contains methods for
running, calculating results, plotting, etc.).

The structure of the ``covasim`` folder is as follows, in the order in which the modules are imported, building from most fundamental to most complex:

* ``version.py``: Version, date, and license information.
* ``requirements.py``: A simple module to check that imports succeeded, and turn off features if they didn't.
* ``utils.py``: Functions for choosing random numbers, many based on Numba, plus other helper functions.
* ``misc.py``: Miscellaneous helper functions.
* ``defaults.py``: The default colors, plots, etc. used by Covasim.
* ``plotting.py``: Plotting scripts, including Plotly graphs for the webapp (used in other Covasim classes, and hence defined first).
* ``base.py``: The ``ParsObj`` class, the fundamental class used in Covasim, plus basic methods of the ``BaseSim`` and ``BasePeople`` classes, and associated functions.
* ``parameters.py``: Functions for creating the parameters dictionary and loading the input data.
* ``people.py``: The ``People`` class, for handling updates of state for each person.
* ``population.py``: Functions for creating populations of people, including age, contacts, etc.
* ``interventions.py``: The ``Intervention`` class, for adding interventions and dynamically modifying parameters, and classes for each of the specific interventions derived from it.
* ``sim.py``: The ``Sim`` class, which performs most of the heavy lifting: initializing the model, running, and plotting.
* ``run.py``: Functions for running simulations (e.g. parallel runs and the ``Scenarios`` and ``MultiSim`` classes).
* ``analysis.py``: The ``Analyzers`` class (for performing analyses on the sim while it's running), the ``Fit`` class (for calculating the fit between the model and the data), the ``TransTree`` class, and other classes and functions for analyzing simulations.


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


Bin
---

This folder contains a command-line interface (CLI) version of Covasim; example usage::

  covasim --pars "{pop_size:20000, pop_infected:1, n_days:360, rand_seed:1}"

Note: the CLI is currently not compatible with Windows. You will need to add
this folder to your path to run from other folders. See the `bin README`_ for more information.

.. _bin README: ./bin


Data
----

Scripts to automatically scrape data (including demographics and COVID epidemiology data),
and the data files themselves (which are not part of the repository). See the `data README`_ for more information.

.. _data README: ./data


Docker
------

This folder contains the ``Dockerfile`` and other files that allow Covasim to be
run as a webapp via Docker. See the `Docker README`_ for more information.

.. _Docker README: ./docker


Examples
--------

This folder contains demonstrations of simple Covasim usage, including an early application of Covasim to the Diamond Princess cruise ship. See the `examples README`_ for more information.

.. _examples README: ./examples


WandB
~~~~~

Utilities for hyperparameter sweeps, using `Weights and Biases`_. See the `Weights and Biases README`_ for more information.

.. _Weights and Biases: https://www.wandb.com/
.. _Weights and Biases  README: https://github.com/InstituteforDiseaseModeling/covasim/tree/master/examples/wandb


Licenses
--------

Licensing information and legal notices.


Tests
-----

Integration, development, and unit tests. While not (yet) beautifully curated, these folders contain many usage examples. See the `tests README`_ for more information.

.. _tests README: ./tests


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