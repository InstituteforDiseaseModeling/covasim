"""
Tests of simulation parameters from
../../covasim/README.md
"""
import unittest
import pytest

from unittest_support_classes import CovaSimTest, TestProperties

MortKeys = TestProperties.ParameterKeys.MortalityKeys
DProgKeys = TestProperties.ParameterKeys.ProgressionKeys
TransKeys = TestProperties.ParameterKeys.TransmissionKeys
TSimKeys = TestProperties.ParameterKeys.SimulationKeys
ResKeys = TestProperties.ResultsDataKeys


pytest.skip("Requires update (regression issue)", allow_module_level=True)


class DiseaseMortalityTests(CovaSimTest):
    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        super().tearDown()
        pass

    def test_default_cfr_one(self):
        """
        Infect lots of people with cfr one and short time to die
        duration. Verify that everyone dies, no recoveries.
        """
        self.set_smallpop_hightransmission()
        default_cfr_one = {
            MortKeys.default_cfr: 1.0
        }
        self.run_sim(default_cfr_one)
        recoveries_channel = self.get_full_result_channel(
            ResKeys.recovered_at_timestep_WHAT
        )
        recoveries_at_timestep_channel = self.get_full_result_channel(
            ResKeys.recovered_at_timestep
        )
        recoveries_cumulative_channel = self.get_full_result_channel(
            ResKeys.recovered_cumulative
        )
        recovery_channels = [
            recoveries_channel,
            recoveries_at_timestep_channel,
            recoveries_cumulative_channel
        ]
        for c in recovery_channels:
            for t in range(len(c)):
                self.assertEqual(0, c[t],
                                 msg="There should be no recoveries"
                                     "with cfr 1.0. Channel {c} had "
                                     "bad data at t: {t}")
                pass
            pass
        cumulative_deaths = self.get_day_final_channel_value(
            ResKeys.deaths_cumulative
        )
        self.assertGreaterEqual(cumulative_deaths, 200,
                                msg="Should be lots of deaths")
        pass

    def test_default_cfr_zero(self):
        """
        Infect lots of people with cfr zero and short time to die
        duration. Verify that no one dies.
        Depends on default_cfr_one
        """
        self.set_smallpop_hightransmission()
        default_cfr_one = {
            MortKeys.default_cfr: 0.0
        }
        self.run_sim(default_cfr_one)
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
                self.assertEqual(0, c[t],
                                 msg="There should be no deaths"
                                     "with cfr 0.0. Channel {c} had "
                                     "bad data at t: {t}")
                pass
            pass
        cumulative_recoveries = self.get_day_final_channel_value(
            ResKeys.recovered_cumulative
        )
        self.assertGreaterEqual(cumulative_recoveries, 200,
                                msg="Should be lots of recoveries")
        pass

    def test_default_cfr_scaling(self):
        """
        Infect lots of people with cfr zero and short time to die
        duration. Verify that no one dies.
        Depends on default_cfr_one
        """
        self.set_smallpop_hightransmission()
        sim_dur = 60
        end_sample_size = 10 # last few days most interesting
        cfrs = [0.01, 0.05, 0.10, 0.15]
        old_ratio_sum = 0
        old_cumulative_deaths = 0
        for cfr in cfrs:
            default_cfr = {
                MortKeys.default_cfr: cfr,
                TSimKeys.number_simulated_days: sim_dur
            }
            self.run_sim(default_cfr)
            deaths_at_timestep_channel = self.get_full_result_channel(
                ResKeys.deaths_daily
            )
            recoveries_at_timestep_channel = self.get_full_result_channel(
                ResKeys.recovered_at_timestep
            )
            new_ratio = []
            for x in range(end_sample_size, len(deaths_at_timestep_channel)):
                new_ratio.append(deaths_at_timestep_channel[x]/
                                 recoveries_at_timestep_channel[x])
                pass
            self.assertGreater(sum(new_ratio), old_ratio_sum,
                               msg="As cfr increases, ratio should increase")
            cumulative_deaths = self.get_day_final_channel_value(
                ResKeys.deaths_cumulative
            )
            self.assertGreaterEqual(cumulative_deaths, old_cumulative_deaths,
                                    msg="Should be more deaths with higer ratio")
            old_cumulative_deaths = cumulative_deaths
            old_ratio_sum = sum(new_ratio)
        pass

    @unittest.skip("P1")
    def test_time_to_die_duration(self):
        pass

    @unittest.skip("P0")
    def test_time_do_die_std(self):
        pass

    @unittest.skip("P3")
    def test_cfr_by_age(self):
        pass

    # TODO: Define these as per parameter definitions