"""
Tests of simulation parameters from
../../covasim/README.md
"""

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
        self.assertEqual(prev_exposed, Hightrans.pop_infected,
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