"""
Tests of simulation parameters from
../../covasim/README.md
"""

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
        prob_dict = {
            DProgKeys.ProbabilityKeys.RelativeProbKeys.crt_to_death_probability: 0.0
        }
        self.set_simulation_prognosis_probability(prob_dict)
        self.run_sim()
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
                                     f" with critical to death probability 0.0. Channel {c} had"
                                     f" bad data at t: {t}")
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
        total_agents = 500
        self.set_everyone_is_going_to_die(num_agents=total_agents)
        death_probs = [0.01, 0.05, 0.10, 0.15]
        old_cumulative_deaths = 0
        for death_prob in death_probs:
            prob_dict = {
                DProgKeys.ProbabilityKeys.RelativeProbKeys.crt_to_death_probability: death_prob
            }
            self.set_simulation_prognosis_probability(prob_dict)
            self.run_sim()
            deaths_at_timestep_channel = self.get_full_result_channel(
                ResKeys.deaths_daily
            )
            recoveries_at_timestep_channel = self.get_full_result_channel(
                ResKeys.recovered_at_timestep
            )
            cumulative_deaths = self.get_day_final_channel_value(
                ResKeys.deaths_cumulative
            )
            self.assertGreaterEqual(cumulative_deaths, old_cumulative_deaths,
                                    msg="Should be more deaths with higer ratio")
            old_cumulative_deaths = cumulative_deaths
        pass

