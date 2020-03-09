'''
Set the parameters for COVID-ABM.
'''

import pylab as pl
from datetime import datetime


__all__ = ['make_pars', 'get_age_sex']


def make_pars():
    ''' Set parameters for the simulation '''
    pars = {}

    # Simulation parameters
    pars['n']          = 20000 # Estimate
    pars['n_infected'] = 10 # Asked for 1000 in Seattle's population
    pars['day_0']      = datetime(2020, 3, 9) # Start day of the epidemic
    pars['n_days']     = 56 # How many days to simulate -- 8 weeks
    pars['seed']       = 1 # Random seed, if None, don't reset
    pars['verbose']    = 1 # Whether or not to display information during the run -- options are 0 (silent), 1 (default), 2 (everything)
    pars['scale']      = 100 # Factor by which to scale results

    # Epidemic parameters
    pars['r_contact']      = 2.5/(10*20) # Probability of infection per contact, estimated
    pars['contacts']       = 20 # Number of contacts per guest per day, estimated
    pars['incub']          = 5.0 # Incubation period, in days, estimated
    pars['incub_std']      = 1.0 # Standard deviation of the serial interval, estimated
    pars['dur']            = 10 # Duration of infectiousness, from https://www.nejm.org/doi/full/10.1056/NEJMc2001737
    pars['dur_std']        = 3 # Variance in duration
    pars['sensitivity']    = 1.0 # Probability of a true positive, estimated
    pars['symptomatic']    = 5 # Increased probability of testing someone symptomatic, estimated

    # Events
    pars['quarantine']       = 0  # Day on which quarantine took effect
    pars['unquarantine']     = -1  # Day on which quarantine took effect
    pars['quarantine_eff']   = 1.00 # Change in transmissibility due to quarantine, estimated

    return pars


def get_age_sex(min_age=0, max_age=99, age_mean=40, age_std=20):
    '''
    Define age-sex distributions.
    '''
    sex = pl.randint(2) # Define female (0) or male (1) -- evenly distributed
    age = pl.normal(age_mean, age_std) # Define age distribution for the crew and guests
    age = pl.median([min_age, age, max_age]) # Normalize
    return age, sex



