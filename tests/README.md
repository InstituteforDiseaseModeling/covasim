# Tests

## pytest

`pytest` will automatically run all the tests in the folder. Just type `pytest` and it will run everything beginning `test_`.

## Coverage and unit tests

To run code coverage on the unit tests:
1) pip install coverage
2) coverage run -m unittest covid_abm_unittests covid_seattle_unittests
3) coverage html

Then open the htmlcov directory and open index.html in a browser
