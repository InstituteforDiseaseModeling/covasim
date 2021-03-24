"""
Tests of simulation parameters from
../../covasim/README.md
"""

import covasim as cv
import unittest
from unittest_support_classes import CovaTest, TProps

DProgKeys = TProps.ParKeys.ProgKeys
TransKeys = TProps.ParKeys.TransKeys
TSimKeys  = TProps.ParKeys.SimKeys
ResKeys   = TProps.ResKeys


class DiseaseMortalityTests(CovaTest):
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
        pop_size = 200
        n_days = 90
        sim = cv.Sim(pop_size=pop_size, pop_infected=pop_size, n_days=n_days)
        for key in ['rel_symp_prob', 'rel_severe_prob', 'rel_crit_prob', 'rel_death_prob']:
            sim[key] = 1e6
        sim.run()
        assert sim.summary.cum_deaths == pop_size

    def test_default_death_prob_zero(self):
        """
        Infect lots of people with cfr zero and short time to die
        duration. Verify that no one dies.
        Depends on default_cfr_one
        """
        total_agents = 500
        self.set_everyone_is_going_to_die(num_agents=total_agents)
        prob_dict = {
            DProgKeys.ProbKeys.RelProbKeys.crt_to_death_probability: 0.0
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
            prob_dict = {DProgKeys.ProbKeys.RelProbKeys.crt_to_death_probability: death_prob}
            self.set_simulation_prognosis_probability(prob_dict)
            self.run_sim()
            cumulative_deaths = self.get_day_final_channel_value(ResKeys.deaths_cumulative)
            self.assertGreaterEqual(cumulative_deaths, old_cumulative_deaths, msg="Should be more deaths with higer ratio")
            old_cumulative_deaths = cumulative_deaths
        pass

# Run unit tests if called as a script
if __name__ == '__main__':
    unittest.main()