import coverage
import unittest
loader = unittest.TestLoader()
cov = coverage.Coverage(source=["covasim.cova_seattle","covasim.cova_base"])
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

# Now pytest tests
import test_age_structure
test_age_structure.test_age_structure(use_popdata=False)

# # These aren't currently working
# import test_sim
# test_sim.test_parsobj()

import test_utils
test_utils.test_rand()
test_utils.test_poisson()
test_utils.test_choose_people()
test_utils.test_choose_people_weighted()

cov.stop()
cov.save()
cov.html_report()