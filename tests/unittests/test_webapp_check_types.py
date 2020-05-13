'''
Test the webapp - check types that would break the InfoHub integration with the covasim webapp
'''

import sciris as sc
import covasim.webapp as cw
import unittest
import json
import datetime


class WebAppInfoHubIntegrationTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        super().tearDown()
        pass

    def test_run_sim_graph_x_axis_type(self):
        # Validate x_axis value type should be datetime format now
        sc.heading('Testing run sim graph x_axis type')

        pars = cw.get_defaults(die=True)
        output = cw.run_sim(sim_pars=pars['sim_pars'], epi_pars=pars['epi_pars'], die=True)
        graphs = output['graphs'][0]['json']

        data = json.loads(graphs)
        x_scatter_first_dt = data['data'][0]['x'][00]
        isinstance(x_scatter_first_dt, type(str))
        assert type(x_scatter_first_dt) == str

        format = "%Y-%m-%d"
        try:
            date_obj = datetime.datetime.strptime(x_scatter_first_dt, format)
            print(date_obj)
        except ValueError:
            print("Incorrect date format.")

    def test_run_sim_graph_y_axis_type(self):
        # Validate y_axis value type should still be float
        sc.heading('Testing run sim graph y_axis type')
        pars = cw.get_defaults(die=True)
        output = cw.run_sim(sim_pars=pars['sim_pars'], epi_pars=pars['epi_pars'], die=True)
        graphs = output['graphs'][0]['json']

        data = json.loads(graphs)
        x_scatter_first_dt = data['data'][0]['y'][00]
        isinstance(x_scatter_first_dt, type(float))
        assert type(x_scatter_first_dt) == float

    def test_run_sim_with_intervention_graph_data(self):
        sc.heading('Testing sim with interventions graph metadata')

        pars = cw.get_defaults(die=True)
        int_pars = {
            'social_distance': [
                {'start': 0, 'end': 90, 'level': 80},
            ],
            'school_closures': [
                {'start': 35, 'end': 90, 'level': 90}
            ],
            'symptomatic_testing': [
                {'start': 45, 'end': 90, 'level': 60}
            ]
        }

        output = cw.run_sim(sim_pars=pars['sim_pars'], epi_pars=pars['epi_pars'], int_pars=int_pars, die=True)
        graphs = output['graphs'][0]['json']

        data = json.loads(graphs)
        infections_name = data['data'][0]['name']
        assert infections_name == 'Cumulative infections'

        diagnoses_name = data['data'][1]['name']
        assert diagnoses_name == 'Cumulative diagnoses'

        recoveries_name = data['data'][2]['name']
        assert recoveries_name == 'Cumulative recoveries'
