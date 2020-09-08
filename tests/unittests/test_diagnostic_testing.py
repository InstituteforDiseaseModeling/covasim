"""
Tests of simulation parameters from
../../covasim/README.md
"""
import unittest

from unittest_support_classes import CovaSimTest, TestProperties

class DiagnosticTestingTests(CovaSimTest):
    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        super().tearDown()
        pass

    @unittest.skip("NYI")
    def test_number_of_daily_tests(self):
        """
        Set sensitivity to 1 and number_daily_tests
        to number_of_agents. Diagnoses should == infections
        """
        pass

    @unittest.skip("NYI")
    def test_daily_test_sensitivity(self):
        """
        With sensitivity to 0, should see no diagnoses
        With sensitivity 0.3 should see more
        With sensitivity 0.7 should see more
        Sensitivity 1 is covered above
        """
        pass

    @unittest.skip("NYI")
    def test_symptomatic_testing_multiplier(self):
        """
        Set whole population to symptomatic, and
        Sensitivity to low positive (0.25)
        See this scale up
        0 no diagnoses
        1.0 more (more than zero)
        2.0 about half (more than 1)
        4.0 about all (more than 2)
        """
        pass

    @unittest.skip("NYI")
    def test_trace_testing_multiplier(self):
        """
        Set number of contacts to a high value
        To see a stronger effect (probably)
        """
        pass