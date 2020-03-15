import coverage
import unittest
loader = unittest.TestLoader()
cov = coverage.Coverage(source=["covid_seattle","covid_abm"])
cov.start()

# First, load and run the unittest tests
from covid_abm_unittests import CovaUnitTests
from covid_seattle_unittests import CovaSeattleUnittests
test_classes_to_run = [CovaUnitTests, CovaSeattleUnittests]

suites_list = []
for tc in test_classes_to_run:
    suite = loader.loadTestsFromTestCase(tc)
    suites_list.append(suite)
    pass

big_suite = unittest.TestSuite(suites_list)
runner = unittest.TextTestRunner()
results = runner.run(big_suite)

cov.stop()
cov.save()
cov.html_report()