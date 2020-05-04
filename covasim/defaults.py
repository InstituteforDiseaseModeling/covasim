'''
Set the defaults across each of the different files.
'''

import numpy as np
import sciris as sc

# Specify all externally visible functions this file defines
__all__ = ['get_colors', 'get_sim_plots', 'get_scen_plots']

#%% Specify what data types to use

default_precision = 32 # Use this by default for speed and memory efficiency
result_float = np.float64 # Always use float64 for results, for simplicity
if default_precision == 32:
    default_float = np.float32
    default_int   = np.int32
elif default_precision == 64:
    default_float = np.float64
    default_int   = np.int64
else:
    raise NotImplementedError


#%% Define all properties of people

class PeopleMeta(sc.prettyobj):
    ''' For storing all the keys relating to a person and people '''

    # Set the properties of a persocn
    person = [
        'uid',         # Any (int or str, usually)
        'age',         # Float
        'sex',         # Int
        'symp_prob',   # Float
        'severe_prob', # Float
        'crit_prob',   # Float
        'death_prob',  # Float
        'rel_trans',   # Float
        'rel_sus',     # Float
    ]

    # Set the states that a person can be in: these are all booleans per person -- used in people.py
    states = [
        'susceptible',
        'exposed',
        'infectious',
        'symptomatic',
        'severe',
        'critical',
        'tested',
        'diagnosed',
        'recovered',
        'dead',
        'known_contact',
        'quarantined',
    ]

    # Set the dates various events took place: these are floats per person -- used in people.py
    dates = [f'date_{state}' for state in states] # Convert each state into a date
    dates.append('date_end_quarantine') # This one is not like the others...

    # Duration of different states: these are floats per person -- used in people.py
    durs = [
        'dur_exp2inf',
        'dur_inf2sym',
        'dur_sym2sev',
        'dur_sev2crit',
        'dur_disease',
    ]

    all_states = person + states + dates + durs


#%% Define other defaults

# A subset of the above states are used for results
result_stocks = {
        'susceptible': 'Number susceptible',
        'exposed':     'Number exposed',
        'infectious':  'Number infectious',
        'symptomatic': 'Number symptomatic',
        'severe':      'Number of severe cases',
        'critical':    'Number of critical cases',
        'diagnosed':   'Number of confirmed cases',
        'quarantined': 'Number in quarantine',
}

# The types of result that are counted as flows -- used in sim.py; value is the label suffix
result_flows = {'infections':  'infections',
                'infectious':  'infectious',
                'tests':       'tests',
                'diagnoses':   'diagnoses',
                'recoveries':  'recoveries',
                'symptomatic': 'symptomatic cases',
                'severe':      'severe cases',
                'critical':    'critical cases',
                'deaths':      'deaths',
                'quarantined': 'quarantined people',
}

# Define these here as well
new_result_flows = [f'new_{key}' for key in result_flows.keys()]
cum_result_flows = [f'cum_{key}' for key in result_flows.keys()]

# Default age data, based on Seattle 2018 census data -- used in population.py
default_age_data = np.array([
            [ 0,  4, 0.0605],
            [ 5,  9, 0.0607],
            [10, 14, 0.0566],
            [15, 19, 0.0557],
            [20, 24, 0.0612],
            [25, 29, 0.0843],
            [30, 34, 0.0848],
            [35, 39, 0.0764],
            [40, 44, 0.0697],
            [45, 49, 0.0701],
            [50, 54, 0.0681],
            [55, 59, 0.0653],
            [60, 64, 0.0591],
            [65, 69, 0.0453],
            [70, 74, 0.0312],
            [75, 79, 0.02016], # Calculated based on 0.0504 total for >=75
            [80, 84, 0.01344],
            [85, 89, 0.01008],
            [90, 99, 0.00672],
            ])


def get_colors():
    '''
    Specify plot colors -- used in sim.py.

    NB, includes duplicates since stocks and flows are named differently.
    '''
    colors = sc.objdict(
        susceptible = '#5e7544',
        infectious  = '#c78f65',
        infections  = '#c75649',
        exposed     = '#c75649', # Duplicate
        tests       = '#aaa8ff',
        diagnoses   = '#8886cc',
        diagnosed   = '#8886cc', # Duplicate
        recoveries  = '#799956',
        recovered   = '#799956', # Duplicate
        symptomatic = '#c1ad71',
        severe      = '#c1981d',
        quarantined = '#5f1914',
        critical    = '#b86113',
        deaths      = '#000000',
        dead        = '#000000', # Duplicate
    )
    return colors


def get_sim_plots():
    ''' Specify which quantities to plot; used in sim.py '''
    plots = sc.odict({
            'Total counts': [
                'cum_infections',
                'cum_diagnoses',
                'cum_recoveries',
            ],
            'Daily counts': [
                'new_infections',
                'new_diagnoses',
                'new_recoveries',
                'new_deaths',
            ],
            'Health outcomes': [
                'cum_severe',
                'cum_critical',
                'cum_deaths',
            ]
    })
    return plots


def get_scen_plots():
    ''' Default scenario plots -- used in run.py '''
    plots = sc.odict({
        'Cumulative infections': [
            'cum_infections',
        ],
        'Number of people currently infectious': [
            'n_infectious',
        ],
        'Number of people requiring hospitalization': [
            'n_severe',
        ]
    })
    return plots

