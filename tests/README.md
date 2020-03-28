# Tests

## pytest

`pytest` will automatically run all the tests in the folder. Just type `pytest` and it will run everything beginning `test_`. You can also run `run_tests`.

## Coverage and unit tests

To run code coverage on the unit tests, you can just type `run_coverage`. If that doesn't work,
try the following:

1. `pip install coverage`
2. `coverage run -m unittest unittest_* test_*`
3. `coverage html`

Then open the htmlcov directory and open index.html in a browser.
