"""
Tests of simulation parameters from
../../covasim/README.md
"""
import unittest

from unittest_support_classes import CovaSimTest, TestProperties

TPKeys = TestProperties.ParameterKeys.SimulationKeys
ResKeys = TestProperties.ResultsDataKeys

class SimulationParameterTests(CovaSimTest):
    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        super().tearDown()
        pass

    def test_population_size(self):
        """
        Set population size to vanilla (1234)
        Run sim for one day and check outputs

        Depends on run default simulation
        """
        self.set_microsim()
        TPKeys = TestProperties.ParameterKeys.SimulationKeys
        pop_10_one_day = {
            TPKeys.population_scaling_factor: 1,
            TPKeys.number_simulated_days: 1,
            TPKeys.number_agents: 10,
            TPKeys.initial_infected_count: 0
        }
        pop_123_one_day = {
            TPKeys.population_scaling_factor: 1,
            TPKeys.number_simulated_days: 1,
            TPKeys.number_agents: 123,
            TPKeys.initial_infected_count: 0
        }
        pop_1234_one_day = {
            TPKeys.population_scaling_factor: 1,
            TPKeys.number_simulated_days: 1,
            TPKeys.number_agents: 1234,
            TPKeys.initial_infected_count: 0
        }
        self.run_sim(pop_10_one_day)
        pop_10_pop = self.get_day_zero_channel_value()
        self.run_sim(pop_123_one_day)
        pop_123_pop = self.get_day_zero_channel_value()
        self.run_sim(pop_1234_one_day)
        pop_1234_pop = self.get_day_zero_channel_value()
        self.assertEqual(pop_10_pop, pop_10_one_day[TPKeys.number_agents])
        self.assertEqual(pop_123_pop, pop_123_one_day[TPKeys.number_agents])
        self.assertEqual(pop_1234_pop, pop_1234_one_day[TPKeys.number_agents])
        pass

    @unittest.skip("See GH 162")
    def test_population_size_ranges(self):
        """
        Intent is to test zero, negative, and excessively large pop sizes
        """
        self.is_debugging = True
        self.set_microsim()
        pop_zero_one_day = {
            TPKeys.population_scaling_factor: 1,
            TPKeys.number_simulated_days: 1,
            TPKeys.number_agents: 0,
            TPKeys.initial_infected_count: 0
        }
        with self.assertRaises(ValueError) as context:
            self.run_sim(pop_zero_one_day)
            pass
        error_message = str(context.exception)
        self.assertIn("n", error_message) # Not awesome but the parameter is 'n'
        pass

    @unittest.skip("See GH 162")
    def test_negative_infected_count(self):
        """
        Test negative infected count
        """
        self.is_debugging = True
        self.set_smallpop_hightransmission()
        negative_infected_count = {
            TPKeys.population_scaling_factor: 1,
            TPKeys.initial_infected_count: -1
        }
        with self.assertRaises(ValueError) as context:
            self.run_sim(negative_infected_count)
            pass
        error_message = str(context.exception)
        self.assertIn('n_infected', error_message)
        pass

    def test_population_scaling(self):
        """
        Scale population vanilla (x10) compare
        output people vs parameter defined people

        Depends on population_size
        """
        self.set_microsim()
        scale_1_one_day = {
            TPKeys.population_scaling_factor: 1,
            TPKeys.number_simulated_days: 1
        }
        scale_2_one_day = {
            TPKeys.population_scaling_factor: 2,
            TPKeys.number_simulated_days: 1
        }
        scale_10_one_day = {
            TPKeys.population_scaling_factor: 10,
            TPKeys.number_simulated_days: 1
        }
        self.run_sim(scale_1_one_day)
        scale_1_pop = self.get_day_zero_channel_value()
        self.run_sim(scale_2_one_day)
        scale_2_pop = self.get_day_zero_channel_value()
        self.run_sim(scale_10_one_day)
        scale_10_pop = self.get_day_zero_channel_value()
        self.assertEqual(scale_2_pop, 2 * scale_1_pop)
        self.assertEqual(scale_10_pop, 10 * scale_1_pop)
        pass

    def test_initial_infected_count(self):
        """
        Set a vanilla number of infections (13)
        Run sim for one day and verify correct count
        """
        infected_0_one_day = {
            TPKeys.number_simulated_days: 1,
            TPKeys.population_scaling_factor: 1,
            TPKeys.initial_infected_count: 0
        }
        infected_1_one_day = {
            TPKeys.population_scaling_factor: 1,
            TPKeys.number_simulated_days: 1,
            TPKeys.initial_infected_count: 1
        }
        infected_321_one_day = {
            TPKeys.population_scaling_factor: 1,
            TPKeys.number_simulated_days: 1,
            TPKeys.initial_infected_count: 321
        }
        self.run_sim(infected_0_one_day)
        key = TestProperties.ResultsDataKeys.exposed_at_timestep
        inf_0_pop = self.get_day_zero_channel_value(key)
        self.run_sim(infected_1_one_day)
        inf_1_pop = self.get_day_zero_channel_value(key)
        self.run_sim(infected_321_one_day)
        inf_321_pop = self.get_day_zero_channel_value(key)
        self.assertEqual(inf_0_pop, infected_0_one_day[TPKeys.initial_infected_count])
        self.assertEqual(inf_1_pop, infected_1_one_day[TPKeys.initial_infected_count])
        self.assertEqual(inf_321_pop, infected_321_one_day[TPKeys.initial_infected_count])
        pass

    def test_random_seed(self):
        """
        Run two simulations with the same seed
        and one with a different one. Something
        randomly drawn (number of persons infected
        day 2) is identical in the first two and
        different in the third
        """
        self.set_smallpop_hightransmission()
        seed_1_params = {
            TPKeys.random_seed: 1
        }
        seed_2_params = {
            TPKeys.random_seed: 2
        }
        self.run_sim(seed_1_params)
        infectious_seed_1_v1 = self.get_full_result_channel(
            ResKeys.infectious_at_timestep
        )
        exposures_seed_1_v1 = self.get_full_result_channel(
            ResKeys.exposed_at_timestep
        )
        self.run_sim(seed_1_params)
        infectious_seed_1_v2 = self.get_full_result_channel(
            ResKeys.infectious_at_timestep
        )
        exposures_seed_1_v2 = self.get_full_result_channel(
            ResKeys.exposed_at_timestep
        )
        self.assertEqual(infectious_seed_1_v1, infectious_seed_1_v2,
                         msg=f"With random seed the same, these channels should"
                             f"be identical.")
        self.assertEqual(exposures_seed_1_v1, exposures_seed_1_v2,
                         msg=f"With random seed the same, these channels should"
                             f"be identical.")
        self.run_sim(seed_2_params)
        infectious_seed_2 = self.get_full_result_channel(
            ResKeys.infectious_at_timestep
        )
        exposures_seed_2 = self.get_full_result_channel(
            ResKeys.exposed_at_timestep
        )
        self.assertNotEqual(infectious_seed_1_v1, infectious_seed_2,
                         msg=f"With random seed the different, these channels should"
                             f"be distinct.")
        self.assertNotEqual(exposures_seed_1_v1, exposures_seed_2,
                         msg=f"With random seed the different, these channels should"
                             f"be distinct.")
        pass

    @unittest.skip('Disabled to improve test suite speed')
    def test_timelimit(self):
        """
        Start timer, run a simulation with many
        persons and a very short time limit
        Verify that the simulation exits after time
        limit expired.
        """
        short_time_limit = {
            TPKeys.time_limit: 0.5
        }
        med_time_limit = {
            TPKeys.time_limit: 1.5
        }
        long_time_limit = {
            TPKeys.time_limit: 15.0
        }
        self.run_sim(params_dict=short_time_limit)
        infections_channel_short = self.get_full_result_channel(
            ResKeys.infectious_at_timestep
        )
        self.run_sim(params_dict=med_time_limit)
        infections_channel_med = self.get_full_result_channel(
            ResKeys.infectious_at_timestep
        )
        self.run_sim(params_dict=long_time_limit)
        infections_channel_long = self.get_full_result_channel(
            ResKeys.infectious_at_timestep
        )
        def remove_zeros(channel):
            while 0 in channel:
                channel.remove(0)
                pass
            return channel
        infections_channel_long = remove_zeros(infections_channel_long)
        infections_channel_med = remove_zeros(infections_channel_med)
        infections_channel_short = remove_zeros(infections_channel_short)
        self.assertGreaterEqual(len(infections_channel_long), len(infections_channel_med))
        self.assertGreaterEqual(len(infections_channel_med), len(infections_channel_short))
        if self.is_debugging:
            print(f"Short sim length: {len(infections_channel_short)}")
            print(f"Med sim length: {len(infections_channel_med)}")
            print(f"Long sim length: {len(infections_channel_long)}")
        pass
    pass
