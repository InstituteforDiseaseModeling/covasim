'''
Set the parameters for LEMOD-FP.
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
    pars['n_guests'] = 2666 # From https://www.princess.com/news/notices_and_advisories/notices/diamond-princess-update.html
    pars['n_crew']   = 1045 # Ditto
    pars['day0']     = datetime(2020, 1, 22) # Start day of the epidemic
    pars['n_days']   = 30 # How many days to simulate
    pars['timestep'] = 1 # Timestep in days
    pars['seed']     = 1 # Random seed, if None, don't reset
    pars['verbose']  = True # Whether or not to display information during the run
    
    # Epidemic parameters
    pars['r_contact']      = 0.1 # Probability of infection per contact
    pars['contacts_guest'] = 10  # Number of contacts per guest per day
    pars['contacts_crew']  = 10 #100 # Number of contacts per crew member per day
    pars['incub']          = 6.0 # Incubation period, in days
    pars['incub_std']      = 1.0 # Standard deviation of the serial interval
    pars['protective_eff'] = 0.9 # Efficacy of protective measures (masks, etc.)
    
    return pars


def get_age_sex(is_crew=False, min_age=18, max_age=99, crew_age=35, crew_std=5, guest_age=68, guest_std=8):
    '''
    Define age-sex distributions. Passenger age distribution based on:
        https://www.nytimes.com/reuters/2020/02/12/world/asia/12reuters-china-health-japan.html
        
        "About 80% of the passengers were aged 60 or over [=2130], with 215 in their 80s and 11 in the 90s, 
        the English-language Japan Times newspaper reported."
    '''
    
    # Define female (0) or male (1) -- evenly distributed
    sex = pl.randint(2)
    
    # Define age distribution for the crew and guests
    if is_crew:
        age = pl.normal(crew_age, crew_std)
    else:
        age = pl.normal(guest_age, guest_std)
    
    # Normalize
    age = pl.median([min_age, age, max_age])
        
    return age, sex


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
        
    
    