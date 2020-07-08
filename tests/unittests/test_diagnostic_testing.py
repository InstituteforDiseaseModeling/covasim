"""
Tests of simulation parameters from
../../covasim/README.md
"""
import unittest

from unittest_support_classes import CovaSimTest, TestProperties

AGENTS = 1000 #number of agents
SimKeys = TestProperties.ParameterKeys.SimulationKeys
DiagKeys = TestProperties.ParameterKeys.DiagnosticTestingKeys
ResultsKeys = TestProperties.ResultsDataKeys
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
        test_sensitivity = 1.0
        DiagKeys.number_daily_tests = AGENTS #Is this the correct way to access this?
        params = {
            SimKeys.number_agents: AGENT_COUNT,
            SimKeys.number_simulated_days: 60} #arbitrary
        start_day = 10
        self.set_simulation_parameters(params_dict=params)

        self.intervention_set_test_prob(test_sensitivity=test_sensitivity, start_day=start_day) #Kind of guessing for some of these
        
        self.run_sim()
        # Getting number infected (or is it infectious cant find how to get just first day)
        first_day_infectious_count = self.get_full_result_channel(
            ResultsKeys.infectious_at_timestep
        )[start_day]
        # Getting number diagnosed on first day
        first_day_diagnoses = self.get_full_result_channel(
                    channel=ResultsKeys.diagnoses_at_timestep
                )[start_day]
        self.assertEqual(first_day_infectious_count, first_day_diagnoses,
                         msg="There should be the same number of diagnoses and infections.")
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