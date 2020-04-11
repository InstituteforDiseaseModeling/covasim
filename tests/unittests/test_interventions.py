from unittest_support_classes import CovaSimTest
from unittest_support_classes import TestProperties

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

    def test_brutal_change_beta_intervention(self):
        params = {
            SimKeys.number_agents: 10000,
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
            SimKeys.number_agents: 10000,
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
