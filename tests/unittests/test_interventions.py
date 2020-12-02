from unittest_support_classes import CovaSimTest
from unittest_support_classes import TestProperties
from math import sqrt
import json
import numpy as np

import unittest

AGENT_COUNT = 1000


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
            SimKeys.number_agents: AGENT_COUNT,
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
            SimKeys.number_agents: AGENT_COUNT,
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
            happy_days = range(days[b], days[b + 1])
            for d in happy_days:
                # print(f"DEBUG: looking at happy day {d}")
                self.assertEqual(new_infections_channel[d],
                                 0,
                                 msg=f"expected 0 infections on day {d}, got {new_infections_channel[d]}.")
            infection_days = range(days[b+1], days[b+2])
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
            SimKeys.number_agents: AGENT_COUNT,
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

    def test_change_beta_layers_clustered(self):
        '''
        Suggested alternative implementation:

            import covasim as cv

            # Define the interventions
            days = dict(h=30, s=35, w=40, c=45)
            interventions = []
            for key,day in days.items():
                interventions.append(cv.change_beta(days=day, changes=0, layers=key))

            # Create and run the sim
            sim = cv.Sim(pop_type='hybrid', n_days=60, interventions=interventions)
            sim.run()
            assert sim.results['new_infections'].values[days['c']:].sum() == 0
            sim.plot()
        '''
        self.is_debugging = False
        initial_infected = 10
        seed_list = range(0)
        for seed in seed_list:
            params = {
                SimKeys.random_seed: seed,
                SimKeys.number_agents: AGENT_COUNT,
                SimKeys.number_simulated_days: 60,
                SimKeys.initial_infected_count: initial_infected
            }
            if len(seed_list) > 1:
                self.expected_result_filename = f"DEBUG_{self.id()}_{seed}.json"
            self.set_simulation_parameters(params_dict=params)
            day_of_change = 25
            change_multipliers = [0.0]
            layer_keys = ['c','h','s','w']

            intervention_days = []
            intervention_list = []

            for k in layer_keys: # Zero out one layer at a time
                day_of_change += 5
                self.intervention_set_changebeta(
                    days_array=[day_of_change],
                    multiplier_array=change_multipliers,
                    layers=[k]
                )
                intervention_days.append(day_of_change)
                intervention_list.append(self.interventions)
                self.interventions = None
                pass
            self.interventions = intervention_list
            self.run_sim(population_type='clustered')
            last_intervention_day = intervention_days[-1]
            first_intervention_day = intervention_days[0]
            cum_infections_channel= self.get_full_result_channel(ResultsKeys.infections_cumulative)
            if len(seed_list) > 1:
                messages = []
                if cum_infections_channel[intervention_days[0]-1] < initial_infected:
                    messages.append(f"Before intervention at day {intervention_days[0]}, there should be infections happening.")
                    pass

                if cum_infections_channel[last_intervention_day] < cum_infections_channel[first_intervention_day]:
                    messages.append(f"Cumulative infections should grow with only some layers enabled.")
                    pass

                if cum_infections_channel[last_intervention_day] != cum_infections_channel[-1]:
                    messages.append(f"The cumulative infections at {last_intervention_day} should be the same as at the end.")
                    pass

                if len(messages) > 0:
                    print(f"ERROR: seed {seed}")
                    for m in messages:
                        print(f"\t{m}")
                        pass

            self.assertGreater(cum_infections_channel[intervention_days[0]-1],
                               initial_infected,
                               msg=f"Before intervention at day {intervention_days[0]}, there should be infections happening.")

            self.assertGreater(cum_infections_channel[last_intervention_day],
                               cum_infections_channel[first_intervention_day],
                               msg=f"Cumulative infections should grow with only some layers enabled.")

            self.assertEqual(cum_infections_channel[last_intervention_day],
                             cum_infections_channel[-1],
                             msg=f"with all layers at 0 beta, the cumulative infections at {last_intervention_day}" +
                                 f" should be the same as at the end.")
            pass

    def test_change_beta_layers_random(self):
        self.is_debugging = False
        initial_infected = 10
        seed_list = range(0)
        for seed in seed_list:
            params = {
                SimKeys.random_seed: seed,
                SimKeys.number_agents: AGENT_COUNT,
                SimKeys.number_simulated_days: 60,
                SimKeys.initial_infected_count: initial_infected
            }
            self.set_simulation_parameters(params_dict=params)
            if len(seed_list) > 1:
                self.expected_result_filename = f"DEBUG_{self.id()}_{seed}.json"
            day_of_change = 25
            change_multipliers = [0.0]
            layer_keys = ['a']

            intervention_days = []
            intervention_list = []

            for k in layer_keys: # Zero out one layer at a time
                day_of_change += 5
                self.intervention_set_changebeta(
                    days_array=[day_of_change],
                    multiplier_array=change_multipliers,
                    layers=[k]
                )
                intervention_days.append(day_of_change)
                intervention_list.append(self.interventions)
                self.interventions = None
                pass
            self.interventions = intervention_list
            self.run_sim(population_type='random')
            last_intervention_day = intervention_days[-1]
            cum_infections_channel = self.get_full_result_channel(ResultsKeys.infections_cumulative)
            if len(seed_list) > 1:
                messages = []
                if cum_infections_channel[intervention_days[0]-1] < initial_infected:
                    messages.append(f"Before intervention at day {intervention_days[0]}, there should be infections happening.")
                    pass

                if cum_infections_channel[last_intervention_day] != cum_infections_channel[-1]:
                    messages.append(f"The cumulative infections at {last_intervention_day} should be the same as at the end.")
                    pass

                if len(messages) > 0:
                    print(f"ERROR: seed {seed}")
                    for m in messages:
                        print(f"\t{m}")
                        pass
            self.assertGreater(cum_infections_channel[intervention_days[0]-1],
                               initial_infected,
                               msg=f"Before intervention at day {intervention_days[0]}, there should be infections happening.")
            self.assertEqual(cum_infections_channel[last_intervention_day],
                             cum_infections_channel[intervention_days[0] - 1],
                             msg=f"With all layers at 0 beta, should be 0 infections at {last_intervention_day}.")

    def test_change_beta_layers_hybrid(self):
        self.is_debugging = False
        initial_infected = 10
        seed_list = range(0)
        for seed in seed_list:
            params = {
                SimKeys.random_seed: seed,
                SimKeys.number_agents: AGENT_COUNT,
                SimKeys.number_simulated_days: 60,
                SimKeys.initial_infected_count: initial_infected
            }
            if len(seed_list) > 1:
                self.expected_result_filename = f"DEBUG_{self.id()}_{seed}.json"
            self.set_simulation_parameters(params_dict=params)
            day_of_change = 25
            change_multipliers = [0.0]
            layer_keys = ['c','s','w','h']

            intervention_days = []
            intervention_list = []

            for k in layer_keys: # Zero out one layer at a time
                day_of_change += 5
                self.intervention_set_changebeta(
                    days_array=[day_of_change],
                    multiplier_array=change_multipliers,
                    layers=[k]
                )
                intervention_days.append(day_of_change)
                intervention_list.append(self.interventions)
                self.interventions = None
                pass
            self.interventions = intervention_list
            self.run_sim(population_type='hybrid')
            last_intervention_day = intervention_days[-1]
            first_intervention_day = intervention_days[0]
            cum_infections_channel = self.get_full_result_channel(ResultsKeys.infections_cumulative)
            if len(seed_list) > 1:
                messages = []
                if cum_infections_channel[intervention_days[0]-1] < initial_infected:
                    messages.append(f"Before intervention at day {intervention_days[0]}, there should be infections happening.")
                    pass

                if cum_infections_channel[last_intervention_day] < cum_infections_channel[first_intervention_day]:
                    messages.append(f"Cumulative infections should grow with only some layers enabled.")
                    pass

                if cum_infections_channel[last_intervention_day] != cum_infections_channel[-1]:
                    messages.append(f"The cumulative infections at {last_intervention_day} should be the same as at the end.")
                    pass

                if len(messages) > 0:
                    print(f"ERROR: seed {seed}")
                    for m in messages:
                        print(f"\t{m}")
                        pass
            self.assertGreater(cum_infections_channel[intervention_days[0]-1],
                               initial_infected,
                               msg=f"Before intervention at day {intervention_days[0]}, there should be infections happening.")
            self.assertGreater(cum_infections_channel[last_intervention_day],
                               cum_infections_channel[first_intervention_day],
                               msg=f"Cumulative infections should grow with only some layers enabled.")
            self.assertEqual(cum_infections_channel[last_intervention_day],
                             cum_infections_channel[-1],
                             msg=f"With all layers at 0 beta, the cumulative infections at {last_intervention_day}"
                                 f" should be the same as at the end.")

    def verify_perfect_test_prob(self, start_day, test_delay, test_sensitivity,
                                 target_pop_count_channel,
                                 target_pop_new_channel,
                                 target_test_count_channel=None):
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
                         target_test_count_channel[start_day],
                         msg=f"Should have each of the {target_test_count_channel[start_day]} targets"
                             f" get tested at day {start_day}. Got {new_tests[start_day]} instead.")
        self.assertEqual(new_diagnoses[start_day + test_delay],
                         target_count[start_day],
                         msg=f"Should have each of the {target_count[start_day]} targets "
                             f"get diagnosed at day {start_day + test_delay} with sensitivity {test_sensitivity} "
                             f"and delay {test_delay}. Got {new_diagnoses[start_day + test_delay]} instead.")
        post_test_days = range(start_day + 1, len(new_tests))
        if target_pop_new_channel:
            for d in post_test_days[:test_delay]:
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


    def test_test_prob_perfect_asymptomatic(self):
        '''
        Test that at 1.0 sensitivity, testing 1.0 asymptomatics finds
        all persons who are infectious but not symptomatic
        '''

        self.is_debugging = False
        agent_count = AGENT_COUNT
        params = {
            SimKeys.number_agents: AGENT_COUNT,
            SimKeys.number_simulated_days: 60
        }
        self.set_simulation_parameters(params_dict=params)

        asymptomatic_probability_of_test = 1.0
        test_sensitivity = 1.0
        test_delay = 0
        start_day = 30

        self.intervention_set_test_prob(asymptomatic_prob=asymptomatic_probability_of_test,
                                        test_sensitivity=test_sensitivity,
                                        test_delay=test_delay,
                                        start_day=start_day)
        self.run_sim()
        symptomatic_count_channel = self.get_full_result_channel(
            ResultsKeys.symptomatic_at_timestep
        )
        infectious_count_channel = self.get_full_result_channel(
            ResultsKeys.infectious_at_timestep
        )
        population_channel = [agent_count] * len(symptomatic_count_channel)
        asymptomatic_infectious_count_channel = list(np.subtract(np.array(infectious_count_channel),
                                                      np.array(symptomatic_count_channel)))
        asymptomatic_population_count_channel = list(np.subtract(np.array(population_channel),
                                                                 np.array(symptomatic_count_channel)))

        self.verify_perfect_test_prob(start_day=start_day,
                                      test_delay=test_delay,
                                      test_sensitivity=test_sensitivity,
                                      target_pop_count_channel=asymptomatic_infectious_count_channel,
                                      target_test_count_channel=asymptomatic_population_count_channel,
                                      target_pop_new_channel=None)

    def test_test_prob_perfect_symptomatic(self):
        self.is_debugging = False
        params = {
            SimKeys.number_agents: AGENT_COUNT,
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
                                      target_pop_new_channel=symptomatic_new_channel,
                                      target_test_count_channel=symptomatic_count_channel
                                      )
        pass

    def test_test_prob_perfect_not_quarantined(self):
        self.is_debugging = False
        agent_count = AGENT_COUNT
        params = {
            SimKeys.number_agents: AGENT_COUNT,
            SimKeys.number_simulated_days: 60
        }
        self.set_simulation_parameters(params_dict=params)

        asymptomatic_probability_of_test = 1.0
        symptomatic_probability_of_test = 1.0
        test_sensitivity = 1.0
        test_delay = 0
        start_day = 30

        self.intervention_set_test_prob(asymptomatic_prob=asymptomatic_probability_of_test,
                                        symptomatic_prob=symptomatic_probability_of_test,
                                        test_sensitivity=test_sensitivity,
                                        test_delay=test_delay,
                                        start_day=start_day)
        self.run_sim()
        infectious_count_channel = self.get_full_result_channel(
            ResultsKeys.infectious_at_timestep
        )
        population_channel = [agent_count] * len(infectious_count_channel)

        self.verify_perfect_test_prob(start_day=start_day,
                                      test_delay=test_delay,
                                      test_sensitivity=test_sensitivity,
                                      target_pop_count_channel=infectious_count_channel,
                                      target_test_count_channel=population_channel,
                                      target_pop_new_channel=None)
        pass

    def test_test_prob_sensitivity(self, subtract_today_recoveries=False):
        self.is_debugging = False
        seed_list = range(0)
        error_seeds = {}
        for seed in seed_list:
            params = {
                SimKeys.random_seed: seed,
                SimKeys.number_agents: AGENT_COUNT,
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
                )[start_day]
                target_count = self.get_full_result_channel(
                    channel=ResultsKeys.symptomatic_at_timestep
                )[start_day]
                if subtract_today_recoveries:
                    recoveries_today = self.get_full_result_channel(
                        channel=ResultsKeys.recovered_at_timestep
                    )[start_day]
                    target_count = target_count - recoveries_today
                ideal_diagnoses = target_count * sensitivity

                standard_deviation = sqrt(sensitivity * (1 - sensitivity) * target_count)
                # 99.7% confidence interval
                min_tolerable_diagnoses = ideal_diagnoses - 3 * standard_deviation
                max_tolerable_diagnoses = ideal_diagnoses + 3 * standard_deviation

                if self.is_debugging:
                    print(f"\tMax: {max_tolerable_diagnoses} \n"
                          f"\tMin: {min_tolerable_diagnoses} \n"
                          f"\tTarget: {target_count} \n"
                          f"\tPrevious day Target: {self.get_full_result_channel(channel=ResultsKeys.symptomatic_at_timestep)[start_day -1 ]} \n"
                          f"\tSensitivity: {sensitivity} \n"
                          f"\tIdeal: {ideal_diagnoses} \n"
                          f"\tActual diagnoses: {first_day_diagnoses}\n")
                    pass

                too_low_message = f"Expected at least {min_tolerable_diagnoses} diagnoses" \
                                  f" with {target_count} symptomatic and {sensitivity}" \
                                  f" sensitivity. Got {first_day_diagnoses} diagnoses," \
                                  f" which is too low."
                too_high_message = f"Expected no more than {max_tolerable_diagnoses} diagnoses" \
                                   f" with {target_count} symptomatic and {sensitivity}" \
                                   f" sensitivity. Got {first_day_diagnoses} diagnoses, which" \
                                   f" is too high."

                if len(seed_list) > 1:
                    local_errors = {}
                    if first_day_diagnoses + 1 < min_tolerable_diagnoses:
                        local_errors[sensitivity] = (f"LOW: {too_low_message} seed: {seed}")
                    elif first_day_diagnoses - 1 > max_tolerable_diagnoses:
                        local_errors[sensitivity] = (f"HIGH: {too_high_message} seed: {seed}")
                    if len(local_errors) > 0:
                        if seed not in error_seeds:
                            error_seeds[seed] = local_errors
                        else:
                            error_seeds[seed][sensitivity] = local_errors[sensitivity]
                else:
                    self.assertGreaterEqual(first_day_diagnoses + 1, min_tolerable_diagnoses,
                                            msg=too_low_message)
                    self.assertLessEqual(first_day_diagnoses - 1, max_tolerable_diagnoses,
                                         msg=too_high_message)
                pass
            pass
        if len(seed_list) > 1:
            with open(f"DEBUG_test_prob_sensitivity_sweep.json",'w') as outfile:
                json.dump(error_seeds, outfile, indent=4)
                pass
            acceptable_losses = len(seed_list) // 10
            self.assertLessEqual(len(error_seeds),
                                 acceptable_losses,
                                 msg=error_seeds)

    def test_test_prob_symptomatic_prob_of_test(self):
        params = {
            SimKeys.number_agents: AGENT_COUNT,
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
            # 99.7% confidence interval
            min_tolerable_tests = ideal_test_count - 3 * standard_deviation
            max_tolerable_tests = ideal_test_count + 3 * standard_deviation
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
    def test_brutal_contact_tracing(self):
        params = {
            SimKeys.number_agents: AGENT_COUNT,
            SimKeys.number_simulated_days: 55
        }
        self.set_simulation_parameters(params_dict=params)

        intervention_list = []

        symptomatic_probability_of_test = 1.0
        test_sensitivity = 1.0
        test_delay = 0
        tests_start_day = 30
        trace_start_day = 40

        self.intervention_set_test_prob(symptomatic_prob=symptomatic_probability_of_test,
                                        test_sensitivity=test_sensitivity,
                                        test_delay=test_delay,
                                        start_day=tests_start_day)
        intervention_list.append(self.interventions)

        trace_probability = 1.0
        trace_delay = 5
        trace_probabilities = {
            'h': trace_probability,
            's': trace_probability,
            'w': trace_probability,
            'c': trace_probability
        }

        trace_delays = {
            'h': trace_delay,
            's': trace_delay,
            'w': trace_delay,
            'c': trace_delay
        }

        self.intervention_set_contact_tracing(start_day=trace_start_day,
                                              trace_probabilities=trace_probabilities,
                                              trace_times=trace_delays)
        intervention_list.append(self.interventions)
        self.interventions = intervention_list
        self.run_sim(population_type='hybrid')
        channel_new_quarantines = self.get_full_result_channel(
            ResultsKeys.quarantined_new
        )
        quarantines_before_tracing = sum(channel_new_quarantines[:trace_start_day])
        quarantines_before_delay_completed = sum(channel_new_quarantines[trace_start_day:trace_start_day + trace_delay])
        quarantines_after_delay = sum(channel_new_quarantines[trace_start_day+trace_delay:])

        self.assertEqual(quarantines_before_tracing, 0,
                         msg="There should be no quarantines until tracing begins.")
        self.assertEqual(quarantines_before_delay_completed, 0,
                         msg="There should be no quarantines until delay expires")
        self.assertGreater(quarantines_after_delay, 0,
                           msg="There should be quarantines after tracing begins")

        pass

    def test_contact_tracing_perfect_school_layer(self):
        self.is_debugging = False
        initial_infected = 10
        params = {
            SimKeys.number_agents: AGENT_COUNT,
            SimKeys.number_simulated_days: 60,
            SimKeys.quarantine_effectiveness: {'c':0.0, 'h':0.0, 'w':0.0, 's':0.0},
            'quar_period': 10,
            SimKeys.initial_infected_count: initial_infected
        }
        self.set_simulation_parameters(params_dict=params)
        sequence_days = [30, 40]
        sequence_interventions = []

        layers_to_zero_beta = ['c','h','w']

        self.intervention_set_test_prob(symptomatic_prob=1.0,
                                        asymptomatic_prob=1.0,
                                        test_sensitivity=1.0,
                                        start_day=sequence_days[1])
        sequence_interventions.append(self.interventions)

        self.intervention_set_changebeta(days_array=[sequence_days[0]],
                                         multiplier_array=[0.0],
                                         layers=layers_to_zero_beta)
        sequence_interventions.append(self.interventions)

        trace_probabilities = {'c': 0, 'h': 0, 'w': 0, 's': 1}
        trace_times         = {'c': 0, 'h': 0, 'w': 0, 's': 0}
        self.intervention_set_contact_tracing(start_day=sequence_days[1],
                                              trace_probabilities=trace_probabilities,
                                              trace_times=trace_times)
        sequence_interventions.append(self.interventions)

        self.interventions = sequence_interventions
        self.run_sim(population_type='hybrid')
        channel_new_infections = self.get_full_result_channel(
            ResultsKeys.infections_at_timestep
        )
        channel_new_tests = self.get_full_result_channel(
            ResultsKeys.tests_at_timestep
        )
        channel_new_diagnoses = self.get_full_result_channel(
            ResultsKeys.diagnoses_at_timestep
        )
        channel_new_quarantine = self.get_full_result_channel(
            ResultsKeys.quarantined_new
        )

        infections_before_quarantine = sum(channel_new_infections[sequence_days[0]:sequence_days[1]])
        infections_after_quarantine  = sum(channel_new_infections[sequence_days[1]:sequence_days[1] + 10])
        if self.is_debugging:
            print(f"Quarantined before, during, three days past sequence:"
                  f" {channel_new_quarantine[sequence_days[0] -1: sequence_days[-1] + 10]}")
            print(f"Tested before, during, three days past sequence:"
                  f" {channel_new_tests[sequence_days[0] -1: sequence_days[-1] + 10]}")
            print(f"Diagnosed before, during, three days past sequence:"
                  f" {channel_new_diagnoses[sequence_days[0] -1: sequence_days[-1] + 10]}")
            print(f"Infections before, during, three days past sequence:"
                  f" {channel_new_infections[sequence_days[0] -1: sequence_days[-1] + 10]}")
            print(f"10 Days after change beta but before quarantine: {infections_before_quarantine} "
                  f"should be less than 10 days after: {infections_after_quarantine}")

        self.assertLess(infections_after_quarantine, infections_before_quarantine,
                        msg=f"10 Days after change beta but before quarantine: {infections_before_quarantine} "
                            f"should be less than 10 days after: {infections_after_quarantine}")


if __name__ == '__main__':
    unittest.main()
