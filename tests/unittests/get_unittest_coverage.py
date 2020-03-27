import coverage
import unittest
loader = unittest.TestLoader()
cov = coverage.Coverage(source=["covasim.base","covasim.model",
                                "covasim.parameters"])
cov.start()

# First, load and run the unittest tests
from unittest_support_classes import TestSupportTests
from simulation_parameter_tests import SimulationParameterTests
from disease_transmission_tests import DiseaseTransmissionTests
from disease_progression_tests import DiseaseProgressionTests
from disease_mortality_tests import DiseaseMortalityTests
from diagnostic_testing_tests import DiagnosticTestingTests

test_classes_to_run = [TestSupportTests,
                       SimulationParameterTests,
                       DiseaseTransmissionTests,
                       DiseaseProgressionTests]

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