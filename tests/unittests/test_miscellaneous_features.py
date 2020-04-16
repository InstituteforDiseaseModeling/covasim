"""

"""

import unittest
from unittest_support_classes import CovaSimTest
from covasim import Sim, parameters
import os

class MiscellaneousFeatureTests(CovaSimTest):
    def setUp(self):
        super().setUp()
        self.sim = Sim()
        self.pars = parameters.make_pars()
        self.is_debugging = False

    def test_xslx_generation(self):
        super().tearDown()
        excel_filename = "DEBUG_test_xslx_generation"
        if os.path.isfile(excel_filename):
            os.unlink(excel_filename)
        self.sim.run(verbose=0)
        self.sim.to_excel(filename=excel_filename)
        pass

    def test_set_pars_invalid_key(self):
        with self.assertRaises(KeyError) as context:
            self.sim['n_infectey'] = 10
            pass
        error_message = str(context.exception)
        self.assertIn('n_infectey', error_message)
        self.assertIn('pop_infected', error_message)
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


