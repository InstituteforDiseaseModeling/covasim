'''
Set the parameters for COVID-ABM.
'''

import pylab as pl
from datetime import datetime


__all__ = ['make_pars', 'get_age_sex', 'load_data']


def make_pars():
    ''' Set parameters for the simulation '''
    pars = {}

    # Simulation parameters
    pars['n']   = 2666 # From https://www.princess.com/news/notices_and_advisories/notices/diamond-princess-update.html
    pars['day_0']      = datetime(2020, 3, 9) # Start day of the epidemic
    pars['n_days']     = 21 # How many days to simulate -- 31 days is until 2020-Feb-20
    pars['seed']       = 1 # Random seed, if None, don't reset
    pars['verbose']    = 1 # Whether or not to display information during the run -- options are 0 (silent), 1 (default), 2 (everything)

    # Epidemic parameters
    pars['r_contact']      = 0.05 # Probability of infection per contact, estimated
    pars['contacts'] = 30 # Number of contacts per guest per day, estimated
    pars['incub']          = 4.0 # Incubation period, in days, estimated
    pars['incub_std']      = 1.0 # Standard deviation of the serial interval, estimated
    pars['dur']            = 12 # Duration of infectiousness, from https://www.nejm.org/doi/full/10.1056/NEJMc2001737
    pars['dur_std']        = 3 # Variance in duration
    pars['sensitivity']    = 1.0 # Probability of a true positive, estimated
    pars['symptomatic']    = 5 # Increased probability of testing someone symptomatic, estimated

    # Events
    pars['quarantine']       = 15  # Day on which quarantine took effect
    pars['quarantine_eff']   = 0.80 # Change in transmissibility due to quarantine, estimated

    return pars


def get_age_sex(min_age=0, max_age=99, age_mean=40, age_std=20):
    '''
    Define age-sex distributions.
    '''
    sex = pl.randint(2) # Define female (0) or male (1) -- evenly distributed
    age = pl.normal(age_mean, age_std) # Define age distribution for the crew and guests
    age = pl.median([min_age, age, max_age]) # Normalize
    return age, sex



