from unittest_support_classes import CovaSimTest
from unittest_support_classes import TestProperties
from math import sqrt
import numpy as np

import unittest

ResultsKeys = TestProperties.ResultsDataKeys
SimKeys = TestProperties.ParameterKeys.SimulationKeys
class InterventionTests(CovaSimTest):
    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        super().tearDown()
        pass

    # region change beta
    def test_brutal_change_beta_intervention(self):
        params = {
            SimKeys.number_agents: 5000,
            SimKeys.number_simulated_days: 60
        }
        self.set_simulation_parameters(params_dict=params)
        day_of_change = 30
        change_days = [day_of_change]
        change_multipliers = [0.0]

        self.intervention_set_changebeta(
            days_array=change_days,
            multiplier_array=change_multipliers
        )
        self.run_sim()
        new_infections_channel = self.get_full_result_channel(
            channel=ResultsKeys.infections_at_timestep
        )
        five_previous_days = range(day_of_change-5, day_of_change)
        for d in five_previous_days:
            self.assertGreater(new_infections_channel[d],
                               0,
                               msg=f"Need to have infections before change day {day_of_change}")
            pass

        happy_days = range(day_of_change + 1, len(new_infections_channel))
        for d in happy_days:
            self.assertEqual(new_infections_channel[d],
                             0,
                             msg=f"expected 0 infections on day {d}, got {new_infections_channel[d]}.")

    def test_change_beta_days(self):
        params = {
            SimKeys.number_agents: 10000,
            SimKeys.number_simulated_days: 60
        }
        self.set_simulation_parameters(params_dict=params)
        # Do a 0.0 intervention / 1.0 intervention on different days
        days =        [ 30,  32,  40,  42,  50]
        multipliers = [0.0, 1.0, 0.0, 1.0, 0.0]
        self.intervention_set_changebeta(days_array=days,
                                         multiplier_array=multipliers)
        self.run_sim()
        new_infections_channel = self.get_full_result_channel(
            channel=ResultsKeys.infections_at_timestep
        )
        five_previous_days = range(days[0] -5, days[0])
        for d in five_previous_days:
            self.assertGreater(new_infections_channel[d],
                               0,
                               msg=f"Need to have infections before first change day {days[0]}")
            pass

        break_days = [0, 2]  # index of "beta to zero" periods
        for b in break_days:
            happy_days = range(days[b] + 1, days[b + 1] + 1)
            for d in happy_days:
                # print(f"DEBUG: looking at happy day {d}")
                self.assertEqual(new_infections_channel[d],
                                 0,
                                 msg=f"expected 0 infections on day {d}, got {new_infections_channel[d]}.")
            infection_days = range(days[b+1] + 1, days[b+2] + 1)
            for d in infection_days:
                # print(f"DEBUG: looking at infection day {d}")
                self.assertGreater(new_infections_channel[d],
                                   0,
                                   msg=f"Expected some infections on day {d}, got {new_infections_channel[d]}")
                pass
            pass
        for d in range (days[-1] + 1, len(new_infections_channel)):
            self.assertEqual(new_infections_channel[d],
                             0,
                             msg=f"After day {days[-1]} should have no infections."
                                 f" Got {new_infections_channel[d]} on day {d}.")

        # verify that every infection day after days[0] is in a 1.0 block
        # verify no infections after 60
        pass

    def test_change_beta_multipliers(self):
        params = {
            SimKeys.number_agents: 5000,
            SimKeys.number_simulated_days: 40
        }
        self.set_simulation_parameters(params_dict=params)
        day_of_change = 20
        change_days = [day_of_change]
        change_multipliers = [1.0, 0.8, 0.6, 0.4, 0.2]
        total_infections = {}
        for multiplier in change_multipliers:
            self.interventions = None

            self.intervention_set_changebeta(
                days_array=change_days,
                multiplier_array=[multiplier]
            )
            self.run_sim(params)
            these_infections = self.get_day_final_channel_value(
                channel=ResultsKeys.infections_cumulative
            )
            total_infections[multiplier] = these_infections
            pass
        for result_index in range(0, len(change_multipliers) - 1):
            my_multiplier = change_multipliers[result_index]
            next_multiplier = change_multipliers[result_index + 1]
            self.assertGreater(total_infections[my_multiplier],
                               total_infections[next_multiplier],
                               msg=f"Expected more infections with multiplier {my_multiplier} "
                                   f"(with {total_infections[my_multiplier]} infections) than {next_multiplier} "
                                   f"(with {total_infections[next_multiplier]} infections)")
    # endregion

    # region test_prob
    def verify_perfect_test_prob(self, start_day, test_delay, test_sensitivity,
                                 target_pop_count_channel,
                                 target_pop_new_channel):
        if test_sensitivity < 1.0:
            raise ValueError("This test method only works with perfect test "
                             f"sensitivity. {test_sensitivity} won't cut it.")
        new_tests = self.get_full_result_channel(
            channel=ResultsKeys.tests_at_timestep
        )
        new_diagnoses = self.get_full_result_channel(
            channel=ResultsKeys.diagnoses_at_timestep
        )
        target_count = target_pop_count_channel
        target_new = target_pop_new_channel
        pre_test_days = range(0, start_day)
        for d in pre_test_days:
            self.assertEqual(new_tests[d],
                             0,
                             msg=f"Should be no testing before day {start_day}. Got some at {d}")
            self.assertEqual(new_diagnoses[d],
                             0,
                             msg=f"Should be no diagnoses before day {start_day}. Got some at {d}")
            pass
        if self.is_debugging:
            print("DEBUGGING")
            print(f"Start day is {start_day}")
            print(f"new tests before, on, and after start day: {new_tests[start_day-1:start_day+2]}")
            print(f"new diagnoses before, on, after start day: {new_diagnoses[start_day-1:start_day+2]}")
            print(f"target count before, on, after start day: {target_count[start_day-1:start_day+2]}")
            pass
        self.assertEqual(new_tests[start_day],
                         target_count[start_day],
                         msg=f"Should have each of the {target_count[start_day]} targets"
                             f" get tested at day {start_day}. Got {new_tests[start_day]} instead.")
        self.assertEqual(new_diagnoses[start_day + test_delay],
                         target_count[start_day],
                         msg=f"Should have each of the {target_count[start_day]} targets "
                             f"get diagnosed at day {start_day + test_delay} with sensitivity {test_sensitivity} "
                             f"and delay {test_delay}. Got {new_diagnoses[start_day + test_delay]} instead.")
        post_test_days = range(start_day + 1, len(new_tests))
        if target_pop_new_channel:
            for d in post_test_days:
                symp_today = target_new[d]
                diag_today = new_diagnoses[d + test_delay]
                test_today = new_tests[d]

                self.assertEqual(symp_today,
                                 test_today,
                                 msg=f"Should have each of the {symp_today} newly symptomatics get"
                                     f" tested on day {d}. Got {test_today} instead.")
                self.assertEqual(symp_today,
                                 diag_today,
                                 msg=f"Should have each of the {symp_today} newly symptomatics get"
                                     f" diagnosed on day {d + test_delay} with sensitivity {test_sensitivity}."
                                     f" Got {test_today} instead.")
            pass
        pass

    @unittest.skip("Don't know how to calc asymptomatic yet")
    def test_test_prob_perfect_asymptomatic(self):
        self.is_debugging = True
        params = {
            SimKeys.number_agents: 5000,
            SimKeys.number_simulated_days: 60
        }
        self.set_simulation_parameters(params_dict=params)

        asymptomatic_probability_of_test = 1.0
        test_sensitivity = 1.0
        test_delay = 2
        start_day = 30

        self.intervention_set_test_prob(asymptomatic_prob=asymptomatic_probability_of_test,
                                        test_sensitivity=test_sensitivity,
                                        test_delay=test_delay,
                                        start_day=start_day)
        self.run_sim()
        symptomatic_count_channel = self.get_full_result_channel(
            ResultsKeys.symptomatic_at_timestep
        )
        population_channel = [5000] * len(symptomatic_count_channel)
        susceptible_count_channel = self.get_full_result_channel(
            ResultsKeys.susceptible_at_timestep
        )
        asymptomatic_count_channel = list(np.subtract(population_channel,
                                                      symptomatic_count_channel))
        asymptomatic_count_channel = list(np.subtract(asymptomatic_count_channel,
                                                      ))
        self.verify_perfect_test_prob(start_day=start_day,
                                      test_delay=test_delay,
                                      test_sensitivity=test_sensitivity,
                                      target_pop_count_channel=asymptomatic_count_channel,
                                      target_pop_new_channel=None)

    def test_test_prob_perfect_symptomatic(self):
        params = {
            SimKeys.number_agents: 5000,
            SimKeys.number_simulated_days: 60
        }
        self.set_simulation_parameters(params_dict=params)

        symptomatic_probability_of_test = 1.0
        test_sensitivity = 1.0
        test_delay = 0
        start_day = 30

        self.intervention_set_test_prob(symptomatic_prob=symptomatic_probability_of_test,
                                        test_sensitivity=test_sensitivity,
                                        test_delay=test_delay,
                                        start_day=start_day)
        self.run_sim()
        symptomatic_count_channel = self.get_full_result_channel(
            ResultsKeys.symptomatic_at_timestep
        )
        symptomatic_new_channel = self.get_full_result_channel(
            ResultsKeys.symptomatic_new_timestep
        )
        self.verify_perfect_test_prob(start_day=start_day,
                                      test_delay=test_delay,
                                      test_sensitivity=test_sensitivity,
                                      target_pop_count_channel=symptomatic_count_channel,
                                      target_pop_new_channel=symptomatic_new_channel
                                      )
        pass

    def test_test_prob_sensitivity(self):
        params = {
            SimKeys.number_agents: 5000,
            SimKeys.number_simulated_days: 31
        }
        self.set_simulation_parameters(params_dict=params)

        symptomatic_probability_of_test = 1.0
        test_sensitivities = [0.9, 0.7, 0.6, 0.2]
        test_delay = 0
        start_day = 30

        for sensitivity in test_sensitivities:
            self.intervention_set_test_prob(symptomatic_prob=symptomatic_probability_of_test,
                                            test_sensitivity=sensitivity,
                                            test_delay=test_delay,
                                            start_day=start_day)
            self.run_sim()
            first_day_diagnoses = self.get_full_result_channel(
                channel=ResultsKeys.diagnoses_at_timestep
            )[start_day + 1]
            target_count = self.get_full_result_channel(
                channel=ResultsKeys.symptomatic_at_timestep
            )[start_day]
            ideal_diagnoses = target_count * sensitivity
            standard_deviation = sqrt(sensitivity * (1 - sensitivity) * target_count)
            # 95% confidence interval
            min_tolerable_diagnoses = ideal_diagnoses - 2 * standard_deviation
            max_tolerable_diagnoses = ideal_diagnoses + 2 * standard_deviation
            if self.is_debugging:
                print(f"Max: {max_tolerable_diagnoses} "
                      f"Min: {min_tolerable_diagnoses} "
                      f"Target: {target_count} "
                      f"Ideal: {ideal_diagnoses} "
                      f"Sensitivity: {sensitivity} "
                      f"Actual diagnoses: {first_day_diagnoses}")
            self.assertGreaterEqual(first_day_diagnoses, min_tolerable_diagnoses,
                                    msg=f"Expected at least {min_tolerable_diagnoses} diagnoses with {target_count}"
                                        f" symptomatic and {sensitivity} sensitivity. Got {first_day_diagnoses}"
                                        f" diagnoses, which is too low.")
            self.assertLessEqual(first_day_diagnoses, max_tolerable_diagnoses,
                                 msg=f"Expected no more than {max_tolerable_diagnoses} diagnoses with {target_count}"
                                     f" symptomatic and {sensitivity} sensitivity. Got {first_day_diagnoses}"
                                     f" diagnoses, which is too high.")
            pass
        pass

    def test_test_prob_symptomatic_prob_of_test(self):
        params = {
            SimKeys.number_agents: 5000,
            SimKeys.number_simulated_days: 31
        }
        self.set_simulation_parameters(params_dict=params)

        symptomatic_probabilities_of_test = [0.9, 0.7, 0.6, 0.2]
        test_sensitivity = 1.0
        test_delay = 0
        start_day = 30

        for s_p_o_t in symptomatic_probabilities_of_test:
            self.intervention_set_test_prob(symptomatic_prob=s_p_o_t,
                                            test_sensitivity=test_sensitivity,
                                            test_delay=test_delay,
                                            start_day=start_day)
            self.run_sim()
            first_day_tests = self.get_full_result_channel(
                channel=ResultsKeys.tests_at_timestep
            )[start_day]
            target_count = self.get_full_result_channel(
                channel=ResultsKeys.symptomatic_at_timestep
            )[start_day]
            ideal_test_count = target_count * s_p_o_t
            standard_deviation = sqrt(s_p_o_t * (1 - s_p_o_t) * target_count)
            # 95% confidence interval
            min_tolerable_tests = ideal_test_count - 2 * standard_deviation
            max_tolerable_tests = ideal_test_count + 2 * standard_deviation
            if self.is_debugging:
                print(f"Max: {max_tolerable_tests} "
                      f"Min: {min_tolerable_tests} "
                      f"Target: {target_count} "
                      f"Ideal: {ideal_test_count} "
                      f"Probability of test: {s_p_o_t} "
                      f"Standard deviation: {standard_deviation} "
                      f"Actual tests: {first_day_tests}")
            self.assertGreaterEqual(first_day_tests, min_tolerable_tests,
                                    msg=f"Expected at least {min_tolerable_tests} tests with {target_count}"
                                        f" symptomatic and {s_p_o_t} sensitivity. Got {first_day_tests}"
                                        f" diagnoses, which is too low.")
            self.assertLessEqual(first_day_tests, max_tolerable_tests,
                                 msg=f"Expected no more than {max_tolerable_tests} tests with {target_count}"
                                     f" symptomatic and {s_p_o_t} sensitivity. Got {first_day_tests}"
                                     f" diagnoses, which is too high.")
            pass
        pass


    # endregion

    # region contact tracing
    @unittest.skip("NYI: Don't know how to use this")
    def test_brutal_contact_tracing(self):
        params = {
            SimKeys.number_agents: 5000,
            SimKeys.number_simulated_days: 60
        }
        self.set_simulation_parameters(params_dict=params)

        symptomatic_probability_of_test = 1.0
        test_sensitivity = 1.0
        test_delay = 0
        start_day = 30

        self.intervention_set_test_prob(symptomatic_prob=symptomatic_probability_of_test,
                                        test_sensitivity=test_sensitivity,
                                        test_delay=test_delay,
                                        start_day=start_day)

        pass

    # endregion