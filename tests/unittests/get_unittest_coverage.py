import coverage
import unittest
loader = unittest.TestLoader()
cov = coverage.Coverage(source=["covasim.base","covasim.interventions",
                                "covasim.parameters","covasim.people",
                                "covasim.population","covasim.misc"])
cov.start()

# First, load and run the unittest tests
from unittest_support_classes import TestSupportTests
from test_miscellaneous_features import MiscellaneousFeatureTests
from test_simulation_parameter import SimulationParameterTests
from test_disease_transmission import DiseaseTransmissionTests
from test_disease_progression import DiseaseProgressionTests
from test_disease_mortality import DiseaseMortalityTests
# from test_diagnostic_testing import DiagnosticTestingTests

test_classes_to_run = [TestSupportTests,
                       SimulationParameterTests,
                       DiseaseTransmissionTests,
                       DiseaseProgressionTests,
                       DiseaseMortalityTests,
                       MiscellaneousFeatureTests]

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