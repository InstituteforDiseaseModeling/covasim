"""
Tests of simulation parameters from
../../covasim/README.md
"""
import unittest

from unittest_support_classes import CovaSimTest, TestProperties
DProgKeys = TestProperties.ParameterKeys.ProgressionKeys
TransKeys = TestProperties.ParameterKeys.TransmissionKeys
TSimKeys = TestProperties.ParameterKeys.SimulationKeys
MortKeys = TestProperties.ParameterKeys.MortalityKeys
ResKeys = TestProperties.ResultsDataKeys

class DiseaseProgressionTests(CovaSimTest):
    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        super().tearDown()
        pass

    def test_exposure_to_infectiousness_delay_deviation_zero(self):
        """
        Configure exposure to infectiousness delay to 1/2 sim
        length, and std_dev to 0. Verify that every n_infected
        at start goes infectious on the same day
        """
        self.is_debugging = False
        self.set_smallpop_hightransmission()
        infectious_day = 30
        initially_infected = 10
        serial_dev_zero = {
            TransKeys.contacts_per_agent: 0, # No transmission
            TSimKeys.number_simulated_days: 60,
            TSimKeys.initial_infected_count: initially_infected,
            DProgKeys.exposed_to_infectious: infectious_day,
            DProgKeys.exposed_to_infectious_std: 0,
            MortKeys.default_cfr: 0.0,
            DProgKeys.infectiousness_duration: 5,
            DProgKeys.infectiousness_duration_std: 1
        }
        self.run_sim(serial_dev_zero)
        infectious_channel = self.get_full_result_channel(
            ResKeys.infectious_at_timestep
        )
        for t in range(0, infectious_day):
            today_infectious = infectious_channel[t]
            self.assertEqual(today_infectious, 0,
                             msg=f"With std_dev 0, there should be no infectious"
                                 f"prior to {infectious_day} + 1. At {t} we had"
                                 f" {today_infectious}.")
            pass
        prev_infectious = infectious_channel[infectious_day]
        self.assertEqual(prev_infectious, initially_infected,
                         msg=f"At {infectious_day} + 1, should have {initially_infected}"
                             f" infectious individuals with std_dev 0. Got {prev_infectious}.")
        for t in range(infectious_day + 1, len(infectious_channel)):
            today_infectious = infectious_channel[t]
            self.assertLessEqual(today_infectious, prev_infectious,
                                 msg="With std_dev 0, after initial infectious conversion,"
                                     " infectiousness should only decline. Not the case"
                                     f" at t: {t}")
            pass
        pass

    @unittest.skip("P1")
    def test_exposure_to_infectiousness_delay_deviation_scaling(self):
        """
        Configure exposure to infectiousness delay to 1/2 sim
        length, and std_dev to 0.5, 1.0, 2.0, 3.0. Verify that
        as std_dev goes up, the first conversion is earlier and
        the last is later.
        Depends on std_dev zero test
        """
        pass

    @unittest.skip("P1")
    def test_exposure_to_infectiousness_delay_max(self):
        """
        Set exposure to infectiousness = simulation length + 1
        Set std_dev to zero. Verify that no one moves to infectious
        Depends on std_dev zero test
        """
        pass

    @unittest.skip("P2")
    def test_exposure_to_infectiousness_delay_scaling(self):
        """
        Set exposure to infectiousness early simulation, mid simulation,
        late simulation. Set std_dev to zero. Verify move to infectiousness
        moves later as delay is longer.
        Depends on delay max test
        """
        pass

    def test_infection_duration_deviation_zero(self):
        """
        Configure all susceptibles to get infected on same early
        timestep, with a single day of infectiousness. Set infection
        duration to "rest of simulation minus one day" and infection
        std_dev of zero. Verify that all of them become recovered on same day
        """
        self.set_smallpop_hightransmission()
        infectious_delay = 2
        initially_infected = 60
        all_agents = 100
        infectious_duration = 58 # this is what's really under test
        serial_dev_zero = {
            TransKeys.contacts_per_agent: 20, # Cartoonishly high
            TransKeys.beta: 10, # Cartoonishly high
            TSimKeys.number_simulated_days: 60 + infectious_delay,
            TSimKeys.initial_infected_count: initially_infected,
            TSimKeys.number_agents: all_agents,
            DProgKeys.exposed_to_infectious: infectious_delay,
            DProgKeys.exposed_to_infectious_std: 0,
            DProgKeys.infectiousness_duration: infectious_duration,
            DProgKeys.infectiousness_duration_std: 0,
            'prog_by_age': False, # HACK: haven't tested this
            'default_death_prob': 0 # HACK: haven't tested this
        }
        self.run_sim(serial_dev_zero)
        infectious_channel = self.get_full_result_channel(
            ResKeys.infectious_at_timestep
        )
        # verify that all "initially infected" are "n_infectious" on infectious_delay day
        self.assertEqual(initially_infected, infectious_channel[infectious_delay],
                         msg="Just making sure, but all initially infected should be infectious here")

        # verify that total population == n_infectious at delay * 2
        prev_infectious = infectious_channel[infectious_delay * 2]
        self.assertEqual(all_agents, prev_infectious,
                         msg="Just making sure, delay + delay is day everyone gets infectious")

        # verify that infectious count remains the same until delay + duration
        for t in range(infectious_delay *2 +1, infectious_delay + infectious_duration):
            today_infectious = infectious_channel[t]
            self.assertEqual(today_infectious, prev_infectious,
                             msg="with duration_std of 0, all persons should remain infectious"
                                 f"for this duration. At t {t} this was not so.")

        # verify that at delay + duration, n_infectious == population - initially infected
        initial_infection_clearing_day = infectious_delay + infectious_duration
        prev_infectious = infectious_channel[initial_infection_clearing_day]
        self.assertEqual(prev_infectious, all_agents - initially_infected,
                         msg=f"Initial infections should clear at {initial_infection_clearing_day},"
                             f" so total infections should be all people minus initial.")

        # verify that at delay + delay + duration, n_infectious == 0
        self.assertEqual(prev_infectious, infectious_channel[initial_infection_clearing_day + 1],
                         msg="Just checking, no more should clear")

        final_infectious_clearing_day = infectious_delay + infectious_delay + infectious_duration
        self.assertEqual(0, infectious_channel[final_infectious_clearing_day],
                         msg="Last infections should clear here.")
        pass

    @unittest.skip("P1")
    def test_infection_duration_deviation_scaling(self):
        """
        Like deviation zero test, but with expected duration midsim.
        Verification like the exposure_to_infectiouness_delay
        deviation scaling test (as std_dev goes up, first conversion
        earlier and last conversion later)
        """
        pass

    def test_infection_duration_scaling(self):
        """
        Make sure that all initial infected cease being infected
        on following day. Std_dev 0 will help here
        """
        self.set_smallpop_hightransmission()
        initially_infected = 60
        all_agents = 100
        setup_params = {
            TransKeys.contacts_per_agent: 20, # Cartoonishly high
            TransKeys.beta: 10, # Cartoonishly high
            TSimKeys.number_simulated_days: 20,
            TSimKeys.initial_infected_count: initially_infected,
            TSimKeys.number_agents: all_agents,
            DProgKeys.exposed_to_infectious_std: 0,
            DProgKeys.infectiousness_duration_std: 0,
            'prog_by_age': False, # HACK: haven't tested this
            'default_death_prob': 0 # HACK: haven't tested this
        }
        self.set_simulation_parameters(setup_params)

        for infectious_duration in [1, 2, 4]: # this is what's really under test
            infectious_delay = infectious_duration + 1
            param_under_test = {
                DProgKeys.infectiousness_duration: infectious_duration,
                DProgKeys.exposed_to_infectious: infectious_delay
            }
            self.run_sim(params_dict=param_under_test)
            infectious_channel = self.get_full_result_channel(
                ResKeys.infectious_at_timestep
            )

            initial_infected_shedding_day = infectious_delay
            initial_susceptible_shedding_day = initial_infected_shedding_day + infectious_delay
            # verify that all "initially infected" are "n_infectious" on infectious_delay day
            self.assertEqual(initially_infected, infectious_channel[initial_infected_shedding_day],
                             msg="Just making sure, but all initially infected should be infectious here")

            # verify that next day, there are no infectious
            self.assertEqual(0, infectious_channel[initial_infected_shedding_day + infectious_duration],
                             msg=f"verify that with infectious_duration {infectious_duration},"
                                 f" and a delay {infectious_delay}, zero infectious people day"
                                 f" after the first batch.")

            # verify that the next day, all the former susceptibles are infectious
            self.assertEqual(all_agents - initially_infected,
                             infectious_channel[initial_susceptible_shedding_day],
                             msg="Just making sure, rest of susceptible become infectious")

            self.assertEqual(0, infectious_channel[initial_susceptible_shedding_day + infectious_duration],
                             msg="verify that after the infectious duration, no more shedders")
        pass
    pass

if __name__ == '__main__':
    unittest.main()