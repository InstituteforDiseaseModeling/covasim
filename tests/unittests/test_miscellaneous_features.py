"""

"""

import unittest
import pandas as pd
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
        excel_filename = "DEBUG_test_xslx_generation.xls"  # NOTE: xslx files won't open in 365 right now
        if os.path.isfile(excel_filename):
            os.unlink(excel_filename)
            pass
        test_infected_value = 31
        params_dict = {
            'pop_infected': test_infected_value
        }
        self.run_sim(params_dict)
        self.sim.to_excel(filename=excel_filename)
        simulation_df = pd.ExcelFile(excel_filename)
        expected_sheets = ['Results','Parameters']
        for sheet in expected_sheets:
            self.assertIn(sheet, simulation_df.sheet_names)
            pass
        params_df = simulation_df.parse('Parameters')
        observed_infected_param = params_df.loc[params_df['Parameter'] == 'pop_infected', 'Value'].values[0]
        self.assertEqual(observed_infected_param, test_infected_value,
                         msg="Should be able to parse the pop_infected parameter from the results sheet")
        results_df = simulation_df.parse('Results')
        observed_day_0_exposed = results_df.loc[results_df['t'] == 0, 'n_exposed'].values[0]
        self.assertEqual(observed_day_0_exposed, test_infected_value,
                         msg="Should be able to parse the day 0 n_exposed value from the results sheet.")
        if not self.is_debugging:
            os.unlink(excel_filename)
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


