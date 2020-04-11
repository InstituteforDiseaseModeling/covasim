=============
Run tests
=============

pytest
======

``pytest`` will automatically run all the tests in the folder. Just type ``pytest`` and it will run everything beginning ``test_``. You can also run ``run_tests``.

Coverage and unit tests
=======================

To run code coverage on the unit tests, you can just type ``run_coverage``. If that doesn't work,
try the following:

#. ``pip install coverage``
#. ``coverage run -m unittest unittest_* test_*``
#. ``coverage html``

Then open the `htmlcov` directory and open index.html in a browser.


For more information about building and running tests in a Docker container,
see :doc:`docker`.