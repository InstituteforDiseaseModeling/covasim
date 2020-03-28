"""
Tests of simulation parameters from
../../covasim/README.md
"""
import unittest

from unittest_support_classes import CovaSimTest, TestProperties

TKeys = TestProperties.ParameterKeys.TransmissionKeys
Hightrans = TestProperties.SpecializedSimulations.Hightransmission

class DiseaseTransmissionTests(CovaSimTest):
    """
    Tests of the parameters involved in transmission
    pre requisites simulation parameter tests
    """

    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        super().tearDown()
        pass

    def test_beta_zero(self):
        """
        Test that with beta at zero, no transmission
        Start with high transmission sim
        """
        self.set_smallpop_hightransmission()
        beta_zero = {
            TKeys.beta: 0
        }
        self.run_sim(beta_zero)
        exposed_today_channel = self.get_full_result_channel(
            TestProperties.ResultsDataKeys.exposed_at_timestep
        )
        prev_exposed = exposed_today_channel[0]
        self.assertEqual(prev_exposed, Hightrans.n_infected,
                         msg="Make sure we have some initial infections")
        for t in range(1, len(exposed_today_channel)):
            today_exposed = exposed_today_channel[t]
            self.assertLessEqual(today_exposed, prev_exposed,
                                    msg=f"The exposure counts should do nothing but decline."
                                        f" At time {t}: {today_exposed} at {t-1}: {prev_exposed}.")
            prev_exposed = today_exposed
            pass

        infections_channel = self.get_full_result_channel(
            TestProperties.ResultsDataKeys.infections_at_timestep
        )
        for t in range(len(infections_channel)):
            today_infectious = infections_channel[t]
            self.assertEqual(today_infectious, 0,
                             msg=f"With beta 0, there should be no infections."
                                 f" At ts: {t} got {today_infectious}.")
            pass
        pass

    @unittest.skip("P1")
    def test_beta_scaling(self):
        """
        Test that as beta increases, transmission is faster
        depends on test_beta_zero
        """
        pass

    def test_contacts_zero(self):
        """
        Test that with zero contacts, no transmission
        depends on test_beta_scaling
        """
        self.set_smallpop_hightransmission()
        beta_zero = {
            TKeys.contacts_per_agent: 0
        }
        self.run_sim(beta_zero)
        exposed_today_channel = self.get_full_result_channel(
            TestProperties.ResultsDataKeys.exposed_at_timestep
        )
        prev_exposed = exposed_today_channel[0]
        self.assertEqual(prev_exposed, Hightrans.n_infected,
                         msg="Make sure we have some initial infections")
        for t in range(1, len(exposed_today_channel)):
            today_exposed = exposed_today_channel[t]
            self.assertLessEqual(today_exposed, prev_exposed,
                                    msg=f"The exposure counts should do nothing but decline."
                                        f" At time {t}: {today_exposed} at {t-1}: {prev_exposed}.")
            prev_exposed = today_exposed
            pass

        infections_channel = self.get_full_result_channel(
            TestProperties.ResultsDataKeys.infections_at_timestep
        )
        for t in range(len(infections_channel)):
            today_infectious = infections_channel[t]
            self.assertEqual(today_infectious, 0,
                             msg=f"With contacts 0, there should be no infections."
                                 f" At ts: {t} got {today_infectious}.")
            pass
        pass

    @unittest.skip("P1")
    def test_contacts_scaling(self):
        """
        Test that as contacts scale 1, 2, 4, 16
        Total transmission scales somewhat as well
        (first 13 transmission days of high transmission sim)
        """
        pass

    @unittest.skip("P1")
    def test_asymptomatic_fraction_zero(self):
        """
        Test that 1.0 aysymptomatic = no n_symptomatic
        Start with high transmission sim
        """
        pass

    @unittest.skip("P2")
    def test_asymptomatic_fraction_scaling(self):
        """
        Test that at 0.1, 0.3, 0.6 we see increasing numbers
        of symptomatics
        depends on test_asymptomatic_zero
        """
        pass

    @unittest.skip("P1")
    def test_asymptomatic_transmission_factor_zero(self):
        """
        Test that at 1.0 symptomatic and 0 trasnmission factor,
        there is no transmission
        Depends on test_asymptomatic_scaling
        """
        pass

    @unittest.skip("P2")
    def test_asymptomatic_transmission_factor_scaling(self):
        """
        Test that at 1.0 symptomatic and 0 trasnmission factor,
        there is no transmission
        Depends on test_asymptomatic_scaling
        """
        pass

    @unittest.skip("P3")
    def test_diagnostic_transmission_factor_does_something(self):
        """
        Test that changing this factor does *something* different
        depends on SimulationParameterTests.test_random_seed
        """
        pass

    @unittest.skip("P3")
    def test_contact_transmission_factor_does_something(self):
        """
        Test that changing this factor does *something* different
        depends on SimulationParameterTests.test_random_seed
        """
        pass

    @unittest.skip("NYI: depends on synthpops testing")
    def test_beta_population_specific(self):
        """
        Depends on synthpops tests
        """
        pass

    @unittest.skip("NYI: depends on synthpops testing")
    def test_contacts_population_specific(self):
        """
        Depends on synthpops tests
        """
        pass
    pass
