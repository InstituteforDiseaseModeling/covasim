'''
Set the parameters for COVID-ABM.
'''

import os
import pylab as pl
import pandas as pd
from datetime import datetime


__all__ = ['make_pars', 'get_age_sex', 'load_data']


def make_pars():
    ''' Set parameters for the simulation '''
    pars = {}

    # Simulation parameters
    pars['scale']      = 1 # Factor by which to scale results ## 100

    pars['n']          = 20e3 # Number ultimately susceptible to CoV
    pars['n_infected'] = 4 # Number of seed cases
    pars['day_0']      = datetime(2020, 2, 17)  #datetime(2020, 2, 10) # Start day of the epidemic 3/5
    pars['n_days']     = 54 # 50 # 25 for calibration, 50 for projections # How many days to simulate
    pars['seed']       = 1 # Random seed, if None, don't reset
    pars['verbose']    = 1 # Whether or not to display information during the run -- options are 0 (silent), 1 (default), 2 (everything)
    pars['usepopdata'] = 0 # Whether or not to load actual population data

    # Epidemic parameters
    # Disease transmission
    pars['beta']           = 0.015 # Beta per symptomatic contact; absolute
    pars['asym_factor']    = 1.0 # Multiply beta by this factor for asymptomatic cases
    pars['contacts']       = 20
    pars['beta_pop']       = {'H': 1.5,  'S': 1.0,   'W': 1.0,  'R': 0.2} # Per-population beta weights; relative
    pars['contacts_pop']   = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 20.0} # default flu-like weights # Number of contacts per person per day, estimated

    # Disease progression
    pars['serial']         = 4.0 # Serial interval: days after exposure before a person can infect others (see e.g. https://www.ncbi.nlm.nih.gov/pubmed/32145466)
    pars['serial_std']     = 1.0 # Standard deviation of the serial interval
    pars['asymptomatic']   = 0.17 # Proportion of asymptomatic cases - estimate based on https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2020.25.10.2000180, #TODO: look for better estimates
    pars['incub']          = 5.0 # Incubation period: days until an exposed person develops symptoms
    pars['incub_std']      = 1.0 # Standard deviation of the incubation period
    pars['dur']            = 8 # Using Mike's Snohomish number
    pars['dur_std']        = 2 # Variance in duration

    # Testing
    pars['sensitivity']    = 1.0 # Probability of a true positive, estimated
    pars['symptomatic']    = 100.0 # Increased probability of testing someone symptomatic, estimated

    # Mortality
    pars['cfr']            = 0.016 # Case fatality rate
    pars['timetodie']      = 21 # Days until death
    pars['timetodie_std']  = 2 # STD

    # Events
    pars['interv_days'] = []# [30, 44]  # Day on which interventions started/stopped
    pars['interv_effs'] = []# [0.1, 10] # Change in transmissibility

    return pars


def get_age_sex(min_age=0, max_age=99, age_mean=40, age_std=15, use_data=True):
    '''
    Define age-sex distributions.
    '''
    if use_data:
        try:
            import synthpops as sp
        except ImportError as E:
            raise ImportError(f'Could not load synthpops; set sim["usepopdata"] = False or install ({str(E)})')
        age, sex = sp.get_seattle_age_sex()
    else:
        sex = pl.randint(2) # Define female (0) or male (1) -- evenly distributed
        age = pl.normal(age_mean, age_std) # Define age distribution for the crew and guests
        age = pl.median([min_age, age, max_age]) # Normalize
    return age, sex


def load_data(filename=None):
    ''' Load data for comparing to the model output '''

    default_datafile = 'oregon-data.xlsx'

    # Handle default filename
    if filename is None:
        cwd = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(cwd, default_datafile)

    # Load data
    raw_data = pd.read_excel(filename)

    # Confirm data integrity and simplify
    cols = ['day', 'date', 'new_tests', 'new_positives', 'new_infections']
    data = pd.DataFrame()
    for col in cols:
        assert col in raw_data.columns, f'Column "{col}" is missing from the loaded data'
    data = raw_data[cols]

    return data



