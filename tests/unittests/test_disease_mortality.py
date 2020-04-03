"""
Tests of simulation parameters from
../../covasim/README.md
"""
import unittest

from unittest_support_classes import CovaSimTest, TestProperties

DProgKeys = TestProperties.ParameterKeys.ProgressionKeys
TransKeys = TestProperties.ParameterKeys.TransmissionKeys
TSimKeys = TestProperties.ParameterKeys.SimulationKeys
ResKeys = TestProperties.ResultsDataKeys

class DiseaseMortalityTests(CovaSimTest):
    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        super().tearDown()
        pass

    def test_default_death_prob_one(self):
        """
        Infect lots of people with cfr one and short time to die
        duration. Verify that everyone dies, no recoveries.
        """
        self.is_debugging = True
        total_agents = 500
        self.set_everyone_is_going_to_die(num_agents=total_agents)
        self.run_sim()
        recoveries_at_timestep_channel = self.get_full_result_channel(
            ResKeys.recovered_at_timestep
        )
        recoveries_cumulative_channel = self.get_full_result_channel(
            ResKeys.recovered_cumulative
        )
        recovery_channels = [
            recoveries_at_timestep_channel,
            recoveries_cumulative_channel
        ]
        for c in recovery_channels:
            for t in range(len(c)):
                self.assertEqual(0, c[t],
                                 msg=f"There should be no recoveries"
                                     f" with death_prob 1.0. Channel {c} had "
                                     f" bad data at t: {t}")
                pass
            pass
        cumulative_deaths = self.get_day_final_channel_value(
            ResKeys.deaths_cumulative
        )
        self.assertEqual(cumulative_deaths, total_agents,
                         msg="Everyone should die")
        pass

    def test_default_death_prob_zero(self):
        """
        Infect lots of people with cfr zero and short time to die
        duration. Verify that no one dies.
        Depends on default_cfr_one
        """
        total_agents = 500
        self.set_everyone_is_going_to_die(num_agents=total_agents)
        all_crit_no_dead = {
            DProgKeys.ProbabilityKeys.crt_to_death_probability: 0.0
        }
        self.run_sim(all_crit_no_dead)
        deaths_at_timestep_channel = self.get_full_result_channel(
            ResKeys.deaths_daily
        )
        deaths_cumulative_channel = self.get_full_result_channel(
            ResKeys.deaths_cumulative
        )
        death_channels = [
            deaths_at_timestep_channel,
            deaths_cumulative_channel
        ]
        for c in death_channels:
            for t in range(len(c)):
                self.assertEqual(c[t], 0,
                                 msg=f"There should be no deaths"
                                     f"with critical to death probability 0.0. Channel {c} had "
                                     f"bad data at t: {t}")
                pass
            pass
        cumulative_recoveries = self.get_day_final_channel_value(
            ResKeys.recovered_cumulative
        )
        self.assertGreaterEqual(cumulative_recoveries, 200,
                                msg="Should be lots of recoveries")
        pass

    def test_default_death_prob_scaling(self):
        """
        Infect lots of people with cfr zero and short time to die
        duration. Verify that no one dies.
        Depends on default_cfr_one
        """
        self.is_debugging = True
        total_agents = 500
        self.set_everyone_is_going_to_die(num_agents=total_agents)
        end_sample_size = 10 # last few days most interesting
        death_probs = [0.01, 0.05, 0.10, 0.15]
        # old_ratio_sum = 0
        old_cumulative_deaths = 0
        for death_prob in death_probs:
            test_config = {
                DProgKeys.ProbabilityKeys.crt_to_death_probability: death_prob
            }
            self.run_sim(test_config)
            deaths_at_timestep_channel = self.get_full_result_channel(
                ResKeys.deaths_daily
            )
            recoveries_at_timestep_channel = self.get_full_result_channel(
                ResKeys.recovered_at_timestep
            )
            # new_ratio = []
            # for x in range(len(deaths_at_timestep_channel) - end_sample_size, len(deaths_at_timestep_channel)):
            #     new_ratio.append(deaths_at_timestep_channel[x]/
            #                      recoveries_at_timestep_channel[x])
            #     pass
            # self.assertGreater(sum(new_ratio), old_ratio_sum,
            #                    msg="As cfr increases, ratio should increase")
            cumulative_deaths = self.get_day_final_channel_value(
                ResKeys.deaths_cumulative
            )
            self.assertGreaterEqual(cumulative_deaths, old_cumulative_deaths,
                                    msg="Should be more deaths with higer ratio")
            old_cumulative_deaths = cumulative_deaths
            # old_ratio_sum = sum(new_ratio)
        pass

    @unittest.skip("P3")
    def test_cfr_by_age(self):
        pass

    # TODO: Define these as per parameter definitions