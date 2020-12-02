=================
Integration tests
=================

This folder contains the core tests for Covasim. Recommended usage is ``./run_tests``. You can also use ``pytest`` to run all the tests in the folder. Description of other scripts included for convenience are below.


check_coverage
--------------

Determine code coverage and create an HTML report. If ``./run_coverage`` doesn't work,
try the following:

1. ``pip install coverage``
2. ``coverage run -m unittest unittest_* test_*``
3. ``coverage html``

Then open the htmlcov directory and open index.html in a browser.


check_everything
----------------

Run integration tests, unit tests, coverage, and build docs.


run_tests
---------

Run all tests, with parallelization, and showing how long each test took.


update_baseline
---------------

The test ``test_baselines.py`` checks to see if results changed unintentionally. If you *intended* to change them, run this script to update the saved results. It also writes default parameter values to the ``../covasim/regression`` folder.