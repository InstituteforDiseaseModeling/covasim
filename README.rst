=======
Covasim
=======

About Covasim
=============

Covasim is a stochastic agent-based simulator designed to be used for COVID-19 (novel coronavirus, SARS-CoV-2) epidemic analyses. These include projections of indicators such as numbers of infections and peak hospital demand. Covasim can also be used to explore the potential impact of different interventions, including social distancing, school closures, testing, contact tracing, quarantine, and vaccination.

The scientific paper describing Covasim is available at http://paper.covasim.org. The recommended citation is:

    Kerr CC, Stuart RM, Mistry D, Abeysuriya RG, Hart G, Rosenfeld R, Selvaraj P, Núñez RC, Hagedorn B, George L, Izzo A, Palmer A, Delport D, Bennette C, Wagner B, Chang S, Cohen JA, Panovska-Griffiths J, Jastrzębski M, Oron AP, Wenger E, Famulare M, Klein DJ (2020). **Covasim: an agent-based model of COVID-19 dynamics and interventions**. *medRxiv* 2020.05.10.20097469; doi: https://doi.org/10.1101/2020.05.10.20097469.

Other papers that have been written using Covasim include:

1. **Controlling COVID-19 via test-trace-quarantine**. Kerr CC, Mistry D, Stuart RM, Rosenfeld R, Hart G, Núñez RC, Selvaraj P, Cohen JA, Abeysuriya RG, George L, Hagedorn B, Jastrzębski M, Fagalde M, Duchin J, Famulare M, Klein DJ (under review). *medRxiv* 2020.07.15.20154765; doi: https://doi.org/10.1101/2020.07.15.20154765.

2. **Determining the optimal strategy for reopening schools, the impact of test and trace interventions, and the risk of occurrence of a second COVID-19 epidemic wave in the UK: a modelling study**. Panovska-Griffiths J, Kerr CC, Stuart RM, Mistry D, Klein DJ, Viner R, Bonnell C (2020). *Lancet Child and Adolescent Health* S2352-4642(20) 30250-9. doi: https://doi.org/10.1016/S2352-4642(20)30250-9.

3. **Modelling the impact of reducing control measures on the COVID-19 pandemic in a low transmission setting**. Scott N, Palmer A, Delport D, Abeysuriya RG, Stuart RM, Kerr CC, Mistry D, Klein DJ, Sacks-Davis R, Heath K, Hainsworth S, Pedrana A, Stoove M, Wilson DP, Hellard M (in press). *Medical Journal of Australia* [`Preprint <https://www.mja.com.au/journal/2020/modelling-impact-reducing-control-measures-covid-19-pandemic-low-transmission-setting>`__, 2 September 2020]; doi: https://doi.org/10.1101/2020.06.11.20127027.

4. **The role of masks in reducing the risk of new waves of COVID-19 in low transmission settings: a modeling study**. Stuart RM, Abeysuriya RG, Kerr CC, Mistry D, Klein DJ, Gray R, Hellard M, Scott N (under review). *medRxiv* 2020.09.02.20186742; doi: https://doi.org/10.1101/2020.09.02.20186742.

5. **Schools are not islands: Balancing COVID-19 risk and educational benefits using structural and temporal countermeasures**. Cohen JA, Mistry D, Kerr CC, Klein DJ (under review). *medRxiv* 2020.09.08.20190942; doi: https://doi.org/10.1101/2020.09.08.20190942.

6. **The potential contribution of face coverings to the control of SARS-CoV-2 transmission in schools and broader society in the UK: a modelling study**. Panovska-Griffiths J, Kerr CC, Waites W, Stuart RM, Mistry D, Foster D, Klein DJ, Viner R, Bonnell C (under review). *medRxiv* 2020.09.28.20202937; doi: https://doi.org/10.1101/2020.09.28.20202937.

(Note: if you have written a paper or report using Covasim, we'd love to know about it! Please write to us `here <mailto:covasim@idmod.org>`__.)

The Covasim webapp is available at http://app.covasim.org, and the repository for it is available `here <https://github.com/institutefordiseasemodeling/covasim_webapp>`__.

Covasim was developed by the `Institute for Disease Modeling <https://idmod.org/>`__, with additional contributions from the `University of Copenhagen <https://www.math.ku.dk/english>`__, the `Burnet Institute <https://www.burnet.edu.au/>`__, `GitHub <https://github.com/>`__, and `Microsoft <https://www.microsoft.com/en-us/ai/ai-for-health-covid-data>`__.

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


Full installation instructions
==============================

If you would rather download the source code rather than using the ``pip`` package, follow these steps:

1.  Clone a copy of the repository. If you intend to make changes to the code, we recommend that you fork it first.

2.  (Optional) Create and activate a virtual environment.

3.  Navigate to the root of the repository and install the Covasim Python package using one of the following options:

    *   For normal installation (recommended)::

          python setup.py develop

    *   To install Covasim and optional dependencies (be aware this may fail since it relies on nonstandard packages)::

          python setup.py develop full

    The module should then be importable via ``import covasim as cv``.


Usage examples
==============

There are several examples in the ``examples`` folder. These can be run as follows:

* ``python examples/simple.py``

  This example creates a figure using default parameter values.

* ``python examples/run_sim.py``

  This shows a slightly more detailed example, including creating an intervention and saving to disk.

* ``python examples/run_scenarios.py``

  This shows a more complex example, including running an intervention scenario, plotting uncertainty, and performing a health systems analysis.

Other examples in that folder are taken from the tutorials.


Module structure
================

All core model code is located in the ``covasim`` subfolder; standard usage is ``import covasim as cv``. The ``data`` subfolder is described below.

The model consists of two core classes: the ``Person`` class (which contains information on health state), and the ``Sim`` class (which contains methods for running, calculating results, plotting, etc.).

The structure of the ``covasim`` folder is as follows, roughly in the order in which the modules are imported, building from most fundamental to most complex:

* ``version.py``: Version, date, and license information.
* ``requirements.py``: A simple module to check that imports succeeded, and turn off features if they didn't.
* ``utils.py``: Functions for choosing random numbers, many based on Numba, plus other helper functions.
* ``misc.py``: Miscellaneous helper functions.
* ``settings.py``: User-customizable options for Covasim (e.g. default font size).
* ``defaults.py``: The default colors, plots, etc. used by Covasim.
* ``parameters.py``: Functions for creating the parameters dictionary and loading the input data.
* ``plotting.py``: Plotting scripts, including Plotly graphs for the webapp (used in other Covasim classes, and hence defined first).
* ``base.py``: The ``ParsObj`` class, the fundamental class used in Covasim, plus basic methods of the ``BaseSim`` and ``BasePeople`` classes, and associated functions.
* ``people.py``: The ``People`` class, for handling updates of state for each person.
* ``population.py``: Functions for creating populations of people, including age, contacts, etc.
* ``interventions.py``: The ``Intervention`` class, for adding interventions and dynamically modifying parameters, and classes for each of the specific interventions derived from it.
* ``sim.py``: The ``Sim`` class, which performs most of the heavy lifting: initializing the model, running, and plotting.
* ``run.py``: Functions for running simulations (e.g. parallel runs and the ``Scenarios`` and ``MultiSim`` classes).
* ``analysis.py``: The ``Analyzers`` class (for performing analyses on the sim while it's running), the ``Fit`` class (for calculating the fit between the model and the data), the ``TransTree`` class, and other classes and functions for analyzing simulations.

The ``data`` folder within the Covasim package contains loading scripts for the epidemiological data in the root ``data`` folder, as well as data on age distributions for different countries and household sizes.



Other folders
=============

Please see the readme in each subfolder for more information.


Bin
---

This folder contains a command-line interface (CLI) version of Covasim; example usage::

  covasim --pars "{pop_size:20000, pop_infected:1, n_days:360, rand_seed:1}"

Note: the CLI is currently not compatible with Windows. You will need to add
this folder to your path to run from other folders.


Data
----

Scripts to automatically scrape data (including demographics and COVID epidemiology data),
and the data files themselves (which are not part of the repository).


Tutorials
---------

This folder contains Jupyter notebooks for nine tutorials that walk you through using Covasim, from absolute basics to advanced topics such as calibration and creating custom populations.


Examples
--------

This folder contains demonstrations of simple Covasim usage, with most examples taken from the tutorials. 


Cruise ship
~~~~~~~~~~~

An early application of Covasim to the Diamond Princess cruise ship.


Calibration
~~~~~~~~~~~

Examples of how to calibrate simulations, including `Optuna`_ (also covered in the tutorial) and `Weights and Biases`_.

.. _Optuna: https://optuna.org/
.. _Weights and Biases: https://www.wandb.com/


Licenses
--------

Licensing information and legal notices.


Tests
-----

Integration, development, and unit tests. While not (yet) beautifully curated, these folders contain many usage examples. See the `tests README`_ for more information.

.. _tests README: ./tests


Disclaimer
==========

The code in this repository was developed by IDM to support our research in disease transmission and managing epidemics. We’ve made it publicly available under the Creative Commons Attribution-ShareAlike 4.0 International License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as contemplated under the Creative Commons Attribution-ShareAlike 4.0 International License. See the contributing and code of conduct READMEs for more information.
