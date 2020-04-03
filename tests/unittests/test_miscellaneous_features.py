"""

"""

import unittest
from unittest_support_classes import CovaSimTest
from covasim import Sim, parameters

class MiscellaneousFeatureTests(CovaSimTest):
    def setUp(self):
        super().setUp()
        self.sim = Sim()
        self.pars = parameters.make_pars()
        self.is_debugging = False

    def test_xslx_generation(self):
        super().tearDown()
        self.sim.run()
        self.sim.to_xlsx()
        pass

    def test_set_pars_invalid_key(self):
        with self.assertRaises(KeyError) as context:
            self.sim['n_infectey'] = 10
            pass
        error_message = str(context.exception)
        self.assertIn('n_infectey', error_message)
        self.assertIn('n_infected', error_message)
        pass

    def test_update_pars_invalid_key(self):
        invalid_key = {
            'dooty_doo': 5
        }
        with self.assertRaises(KeyError) as context:
            self.sim.update_pars(invalid_key)
            pass
        error_message = str(context.exception)
        self.assertIn('dooty_doo', error_message)
        pass

    def test_update_pars_invalid_type(self):
        invalid_key = ('dooty_doo', 5)
        with self.assertRaises(TypeError) as context:
            self.sim.update_pars(invalid_key)
            pass
        error_message = str(context.exception)
        self.assertIn('dict', error_message)
        pass
