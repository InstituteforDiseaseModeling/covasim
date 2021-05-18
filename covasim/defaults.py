'''
Set the defaults across each of the different files.

To change the default precision from 32 bit (default) to 64 bit, use::

    cv.options.set(precision=64)
'''

import numpy as np
import numba as nb
import sciris as sc
from .settings import options as cvo # To set options

# Specify all externally visible functions this file defines -- other things are available as e.g. cv.defaults.default_int
__all__ = ['default_float', 'default_int', 'get_default_colors', 'get_default_plots']


#%% Specify what data types to use

result_float = np.float64 # Always use float64 for results, for simplicity
if cvo.precision == 32:
    default_float = np.float32
    default_int   = np.int32
    nbfloat       = nb.float32
    nbint         = nb.int32
elif cvo.precision == 64: # pragma: no cover
    default_float = np.float64
    default_int   = np.int64
    nbfloat       = nb.float64
    nbint         = nb.int64
else:
    raise NotImplementedError(f'Precision must be either 32 bit or 64 bit, not {cvo.precision}')


#%% Define all properties of people

class PeopleMeta(sc.prettyobj):
    ''' For storing all the keys relating to a person and people '''

    def __init__(self):

        # Set the properties of a person
        self.person = [
            'uid',              # Int
            'age',              # Float
            'sex',              # Float
            'symp_prob',        # Float
            'severe_prob',      # Float
            'crit_prob',        # Float
            'death_prob',       # Float
            'rel_trans',        # Float
            'rel_sus',          # Float
        ]

        # Set the states that a person can be in: these are all booleans per person -- used in people.py
        self.states = [
            'susceptible',
            'naive',
            'exposed',
            'infectious',
            'symptomatic',
            'severe',
            'critical',
            'tested',
            'diagnosed',
            'recovered',
            'known_dead',
            'dead',
            'known_contact',
            'quarantined',
            'vaccinated',
        ]

        # Strain states -- these are ints
        self.strain_states = [
            'exposed_strain',
            'infectious_strain',
            'recovered_strain',
        ]

        # Strain states -- these are ints, by strain
        self.by_strain_states = [
            'exposed_by_strain',
            'infectious_by_strain',
        ]

        # Immune states, by strain
        self.imm_states = [
            'sus_imm',          # Float, by strain
            'symp_imm',         # Float, by strain
            'sev_imm',          # Float, by strain
        ]

        # Neutralizing antibody states, not by strain
        self.nab_states = [
            'prior_symptoms',   # Float
            'init_nab',         # Float, initial neutralization titre relative to convalescent plasma
            'nab',              # Float, current neutralization titre relative to convalescent plasma
        ]

        # Additional vaccination states
        self.vacc_states = [
            'vaccinations',     # Number of doses given per person
            'vaccine_source',   # index of vaccine that individual received
        ]

        # Set the dates various events took place: these are floats per person -- used in people.py
        self.dates = [f'date_{state}' for state in self.states] # Convert each state into a date
        self.dates.append('date_pos_test') # Store the date when a person tested which will come back positive
        self.dates.append('date_end_quarantine') # Store the date when a person comes out of quarantine

        # Duration of different states: these are floats per person -- used in people.py
        self.durs = [
            'dur_exp2inf',
            'dur_inf2sym',
            'dur_sym2sev',
            'dur_sev2crit',
            'dur_disease',
        ]

        self.all_states = self.person + self.states + self.strain_states + self.by_strain_states + self.imm_states + self.nab_states + self.vacc_states + self.dates + self.durs

        # Validate
        self.state_types = ['person', 'states', 'strain_states', 'by_strain_states', 'imm_states', 'nab_states', 'vacc_states', 'dates', 'durs', 'all_states']
        for state_type in self.state_types:
            states = getattr(self, state_type)
            n_states        = len(states)
            n_unique_states = len(set(states))
            if n_states != n_unique_states: # pragma: no cover
                errormsg = f'In {state_type}, only {n_unique_states} of {n_states} state names are unique'
                raise ValueError(errormsg)

        return



#%% Define other defaults

# A subset of the above states are used for results
result_stocks = {
    'susceptible': 'Number susceptible',
    'exposed':     'Number exposed',
    'infectious':  'Number infectious',
    'symptomatic': 'Number symptomatic',
    'severe':      'Number of severe cases',
    'critical':    'Number of critical cases',
    'recovered':   'Number recovered',
    'dead':        'Number dead',
    'diagnosed':   'Number of confirmed cases',
    'known_dead':  'Number of confirmed deaths',
    'quarantined': 'Number in quarantine',
    'vaccinated':  'Number of people vaccinated',
}

result_stocks_by_strain = {
    'exposed_by_strain':    'Number exposed by strain',
    'infectious_by_strain': 'Number infectious by strain',
}

# The types of result that are counted as flows -- used in sim.py; value is the label suffix
result_flows = {
    'infections':   'infections',
    'reinfections': 'reinfections',
    'infectious':   'infectious',
    'symptomatic':  'symptomatic cases',
    'severe':       'severe cases',
    'critical':     'critical cases',
    'recoveries':   'recoveries',
    'deaths':       'deaths',
    'tests':        'tests',
    'diagnoses':    'diagnoses',
    'known_deaths': 'known deaths',
    'quarantined':  'quarantined people',
    'vaccinations': 'vaccinations',
    'vaccinated':   'vaccinated people'
}

result_flows_by_strain = {
    'infections_by_strain': 'infections by strain',
    'infectious_by_strain': 'infectious by strain',
}

result_imm = {
    'pop_nabs': 'Population average nabs',
    'pop_protection': 'Population average protective immunity'
}

# Define new and cumulative flows
new_result_flows = [f'new_{key}' for key in result_flows.keys()]
cum_result_flows = [f'cum_{key}' for key in result_flows.keys()]
new_result_flows_by_strain = [f'new_{key}' for key in result_flows_by_strain.keys()]
cum_result_flows_by_strain = [f'cum_{key}' for key in result_flows_by_strain.keys()]

# Parameters that can vary by strain
strain_pars = [
    'rel_imm_strain',
    'rel_beta',
    'rel_symp_prob',
    'rel_severe_prob',
    'rel_crit_prob',
    'rel_death_prob',
]

# Immunity is broken down according to 3 axes, as listed here
immunity_axes = ['sus', 'symp', 'sev']

# Immunity protection also varies depending on your infection history
immunity_sources = [
    'asymptomatic',
    'mild',
    'severe',
]

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


def get_default_colors():
    '''
    Specify plot colors -- used in sim.py.

    NB, includes duplicates since stocks and flows are named differently.
    '''
    c = sc.objdict()
    c.susceptible           = '#4d771e'
    c.exposed               = '#c78f65'
    c.exposed_by_strain     = '#c75649',
    c.infectious            = '#e45226'
    c.infectious_by_strain  = c.infectious
    c.infections            = '#b62413'
    c.reinfections          = '#732e26'
    c.infections_by_strain  = '#b62413'
    c.tests                 = '#aaa8ff'
    c.diagnoses             = '#5f5cd2'
    c.diagnosed             = c.diagnoses
    c.quarantined           = '#5c399c'
    c.vaccinations          = c.quarantined # TODO: new color
    c.vaccinated            = c.quarantined
    c.recoveries            = '#9e1149'
    c.recovered             = c.recoveries
    c.symptomatic           = '#c1ad71'
    c.severe                = '#c1981d'
    c.critical              = '#b86113'
    c.deaths                = '#000000'
    c.dead                  = c.deaths
    c.known_dead            = c.deaths
    c.known_deaths          = c.deaths
    c.default               = '#000000'
    c.pop_nabs              = '#32733d'
    c.pop_protection        = '#9e1149'
    c.pop_symp_protection   = '#b86113'
    return c


# Define the 'overview plots', i.e. the most useful set of plots to explore different aspects of a simulation
overview_plots = [
    'cum_infections',
    'cum_severe',
    'cum_critical',
    'cum_deaths',
    'cum_known_deaths',
    'cum_diagnoses',
    'new_infections',
    'new_severe',
    'new_critical',
    'new_deaths',
    'new_diagnoses',
    'n_infectious',
    'n_severe',
    'n_critical',
    'n_susceptible',
    'new_tests',
    'n_symptomatic',
    'new_quarantined',
    'n_quarantined',
    'new_vaccinations',
    'new_vaccinated',
    'cum_vaccinated',
    'cum_vaccinations',
    'test_yield',
    'r_eff',
]

overview_strain_plots = [
    'cum_infections_by_strain',
    'new_infections_by_strain',
    'n_infectious_by_strain',
    'cum_reinfections',
    'new_reinfections',
    'pop_nabs',
    'pop_protection',
    'pop_symp_protection',
]

def get_default_plots(which='default', kind='sim', sim=None):
    '''
    Specify which quantities to plot; used in sim.py.

    Args:
        which (str): either 'default' or 'overview'
    '''

    # Default plots -- different for sims and scenarios
    if which in [None, 'default']:

        if 'sim' in kind:
            plots = sc.odict({
                    'Total counts': [
                        'cum_infections',
                        'n_infectious',
                        'cum_diagnoses',
                    ],
                    'Daily counts': [
                        'new_infections',
                        'new_diagnoses',
                    ],
                    'Health outcomes': [
                        'cum_severe',
                        'cum_critical',
                        'cum_deaths',
                        'cum_known_deaths',
                    ],
            })

        elif 'scen' in kind: # pragma: no cover
            plots = sc.odict({
                'Cumulative infections': [
                    'cum_infections',
                ],
                'New infections per day': [
                    'new_infections',
                ],
                'Cumulative deaths': [
                    'cum_deaths',
                    'cum_known_deaths',
                ],
            })

        else:
            errormsg = f'Expecting "sim" or "scens", not "{kind}"'
            raise ValueError(errormsg)

    # Show an overview
    elif which == 'overview': # pragma: no cover
        plots = sc.dcp(overview_plots)

    # Plot absolutely everything
    elif which.lower() == 'all': # pragma: no cover
        plots = sim.result_keys('all')

    # Show an overview plus strains
    elif 'overview' in which and 'strain' in which: # pragma: no cover
        plots = sc.dcp(overview_plots) + sc.dcp(overview_strain_plots)

    # Show default but with strains
    elif 'strain' in which: # pragma: no cover
        plots = sc.odict({
                'Cumulative infections by strain': [
                    'cum_infections_by_strain',
                ],
                'New infections by strain': [
                    'new_infections_by_strain',
                ],
                'Diagnoses': [
                    'cum_diagnoses',
                    'new_diagnoses',
                ],
                'Health outcomes': [
                    'cum_severe',
                    'cum_critical',
                    'cum_deaths',
                ],
        })

    # Plot SEIR compartments
    elif which.lower() == 'seir': # pragma: no cover
        plots = [
            'n_susceptible',
            'n_preinfectious',
            'n_infectious',
            'n_removed',
        ],

    else: # pragma: no cover
        errormsg = f'The choice which="{which}" is not supported: choices are "default", "overview", "all", "strain", "overview-strain", or "seir"'
        raise ValueError(errormsg)

    return plots
