'''
Set the parameters for LEMOD-FP.
'''

import os
from datetime import datetime
import pandas as pd


__all__ = ['make_pars', 'load_data']


def make_pars():
    ''' Set parameters for the simulation '''
    pars = {}

    # Simulation parameters
    pars['n_guests'] = 2666 # From https://www.princess.com/news/notices_and_advisories/notices/diamond-princess-update.html
    pars['n_crew'] = 1045 # Ditto
    pars['day0'] = datetime(2020, 1, 22) # Start day of the epidemic
    pars['n_days'] = 30 # How many days to simulate
    pars['timestep'] = 1 # Timestep in days
    pars['seed'] = 1 # Random seed, if None, don't reset
    pars['verbose'] = True # Whether or not to display information during the run
    
    # Epidemic parameters
    pars['r_contact']      = 0.1 # Probability of infection per contact
    pars['contacts_guest'] = 10  # Number of contacts per guest per day
    pars['contacts_crew']  = 100 # Number of contacts per crew member per day
    pars['incub']          = 6.0 # Incubation period, in days
    pars['incub_std']      = 1.0 # Standard deviation of the serial interval
    pars['protective_eff'] = 0.9 # Efficacy of protective measures (masks, etc.)
    
    return pars


def load_data(filename=None):
    ''' Load data for comparing to the model output '''
    
    # Handle default filename
    if filename is None:
        cwd = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(cwd, 'reported_infections.csv')
    
    # Load data
    raw_data = pd.read_csv(filename)
    
    # Confirm data integrity and simplify
    cols = ['day', 'date', 'new_tests', 'new_infections']
    data = pd.DataFrame()
    for col in cols:
        assert col in raw_data.columns, f'Column "{col}" is missing from the loaded data'
    data = raw_data[cols]
    
    return data
        
    
    