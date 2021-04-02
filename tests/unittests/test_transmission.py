"""
Tests of simulation parameters from
../../covasim/README.md
"""

from unittest_support import CovaTest, TProps

TKeys = TProps.ParKeys.TransKeys
Hightrans = TProps.SpecialSims.Hightransmission

class DiseaseTransmissionTests(CovaTest):
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
            'beta': 0
        }
        self.run_sim(beta_zero)
        exposed_today_ch = self.get_full_result_ch(
            TProps.ResKeys.exposed_at_timestep
        )
        prev_exposed = exposed_today_ch[0]
        self.assertEqual(prev_exposed, Hightrans.pop_infected,
                         msg="Make sure we have some initial infections")
        for t in range(1, len(exposed_today_ch)):
            today_exposed = exposed_today_ch[t]
            self.assertLessEqual(today_exposed, prev_exposed,
                                    msg=f"The exposure counts should do nothing but decline."
                                        f" At time {t}: {today_exposed} at {t-1}: {prev_exposed}.")
            prev_exposed = today_exposed
            pass

        infections_ch = self.get_full_result_ch(
            TProps.ResKeys.infections_at_timestep
        )
        for t in range(len(infections_ch)):
            today_infectious = infections_ch[t]
            self.assertEqual(today_infectious, 0,
                             msg=f"With beta 0, there should be no infections."
                                 f" At ts: {t} got {today_infectious}.")
            pass
        pass