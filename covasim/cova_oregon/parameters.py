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

    pars['n']          = 30000 # Number ultimately susceptible to CoV
    pars['n_infected'] = 3 # Number of seed cases
    pars['day_0']      = datetime(2020, 2, 21)  #datetime(2020, 2, 10) # Start day of the epidemic 3/5
    pars['n_days']     = 50 # 25 for calibration # How many days to simulate
    pars['seed']       = 1 # Random seed, if None, don't reset
    pars['verbose']    = 2 # Whether or not to display information during the run -- options are 0 (silent), 1 (default), 2 (everything)
    pars['usepopdata'] = 0 # Whether or not to load actual population data

    # Epidemic parameters
    pars['beta']           = 0.013 # Beta per contact; absolute
    pars['contacts']       = 20 # Beta per contact; absolute
    pars['beta_pop']       = {'H': 1.5,  'S': 1.0,   'W': 1.0,  'R': 0.2} # Per-population beta weights; relative
    pars['contacts_pop']   = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 20.0} # default flu-like weights # Number of contacts per person per day, estimated
    pars['incub']          = 4.0 # Using Mike's Snohomish number
    pars['incub_std']      = 1.0 # Standard deviation of the serial interval, estimated
    pars['dur']            = 8 # Using Mike's Snohomish number
    pars['dur_std']        = 2 # Variance in duration
    pars['sensitivity']    = 1.0 # Probability of a true positive, estimated
    pars['symptomatic']    = 100.0 # Increased probability of testing someone symptomatic, estimated
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
    cols = ['day', 'date', 'new_tests', 'new_positives']
    data = pd.DataFrame()
    for col in cols:
        assert col in raw_data.columns, f'Column "{col}" is missing from the loaded data'
    data = raw_data[cols]

    return data



