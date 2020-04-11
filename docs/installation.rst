============
Installation
============

The following instructions walk you through installing |Cov_l|. If you want to
run the web UI, see :doc:`webui`. You may also build and test |Cov_s| using
Docker containers (:doc:`docker`).

Requirements
============

|Python_supp|. (Note: Python 2 is not supported.)

We also recommend, but do not require, using Python virtual environments. For
more information, see documentation for `venv`_ or `Anaconda`_.

.. _venv: https://docs.python.org/3/tutorial/venv.html
.. _Anaconda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Quick start guide
=================

Install with ``pip install covasim``. If everything is working, the following Python commands should bring up a plot::

    python
    import covasim as cv
    sim = cv.Sim()
    sim.run()
    sim.plot()



Detailed installation instructions
==================================

#.  Clone a copy of the repository. If you intend to make changes to the code,
    we recommend that you fork it first.

#.  (Optional) Create and activate a virtual environment.

#.  Navigate to the root of the repository and install the |Cov_s| Python package
    using one of the following options:

    *   To install with webapp support (recommended)::

            python setup.py develop

    *   To install as a standalone Python model without webapp support::

            python setup.py develop nowebapp

    *   To install |Cov_s| and optional dependencies (be aware this may fail
        since it relies on private packages), enter::

            python setup.py develop full

    The module should then be importable via ``import covasim``.

.. toctree::

   webui
   docker


