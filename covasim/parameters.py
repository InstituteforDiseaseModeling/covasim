'''
Set the parameters for Covasim.
'''

import pylab as pl
import pandas as pd
from datetime import datetime
import numba as nb
import sciris as sc


__all__ = ['make_pars', 'load_data']


def make_pars():
    '''
    Set parameters for the simulation.

    NOTE: If you update these values or add a new parameter, please update README.md
    in this folder as well.
    '''
    pars = {}

    # Simulation parameters
    pars['scale']      = 1 # Factor by which to scale results -- e.g. 0.6*100 with n=10e3 assumes 60% of a population of 1m

    pars['n']          = 20e3 # Number ultimately susceptible to CoV
    pars['n_infected'] = 10 # Number of seed cases
    pars['start_day']  = datetime(2020, 3, 1) # Start day of the simulation
    pars['n_days']     = 60 # Number of days of run, if end_day isn't used
    pars['seed']       = 1 # Random seed, if None, don't reset
    pars['verbose']    = 1 # Whether or not to display information during the run -- options are 0 (silent), 1 (default), 2 (everything)
    pars['usepopdata'] = 'random' # Whether or not to load actual population data
    pars['timelimit']  = 3600 # Time limit for a simulation (seconds)
    pars['stop_func']  = None # A function to call to stop the sim partway through
    pars['window']     = 7 # Integration window for doubling time and R_eff

    # Disease transmission
    pars['beta']           = 0.015 # Beta per symptomatic contact; absolute
    pars['asym_factor']    = 0.8 # Multiply beta by this factor for asymptomatic cases
    pars['diag_factor']    = 0.0 # Multiply beta by this factor for diganosed cases -- baseline assumes no isolation
    pars['cont_factor']    = 1.0 # Multiply beta by this factor for people who've been in contact with known positives  -- baseline assumes no isolation
    pars['contacts']       = 20 # Estimated number of contacts
    pars['beta_pop']       = {'H': 1.5,  'S': 1.5,   'W': 1.5,  'R': 0.5} # Per-population beta weights; relative
    pars['contacts_pop']   = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 20.0} # default flu-like weights # Number of contacts per person per day, estimated

    # Disease progression
    pars['serial']         = 4.0 # Serial interval: days after exposure before a person can infect others (see e.g. https://www.ncbi.nlm.nih.gov/pubmed/32145466)
    pars['serial_std']     = 1.0 # Standard deviation of the serial interval
    pars['incub']          = 5.0 # Incubation period: days until an exposed person develops symptoms
    pars['incub_std']      = 1.0 # Standard deviation of the incubation period
    pars['dur']            = 8 # Using Mike Famulare's Snohomish number
    pars['dur_std']        = 2 # Variance in duration

    # Mortality and severity
    pars['timetodie']           = 21 # Days until death
    pars['timetodie_std']       = 2 # STD
    pars['prog_by_age']         = True # Whether or not to use age-specific probabilities of prognosis (symptoms/severe symptoms/death)
    pars['default_symp_prob']   = 0.7 # If not using age-specific values: overall proportion of symptomatic cases
    pars['default_severe_prob'] = 0.3 # If not using age-specific values: proportion of symptomatic cases that become severe (default 0.2 total)
    pars['default_death_prob']  = 0.07 # If not using age-specific values: proportion of severe cases that result in death (default 0.02 CFR)

    # Events and interventions
    pars['interventions'] = []  #: List of Intervention instances
    pars['interv_func'] = None # Custom intervention function

    return pars


def load_data(filename):
    ''' Load data for comparing to the model output '''

    # Load data
    raw_data = pd.read_excel(filename)

    # Confirm data integrity and simplify
    cols = ['day', 'date', 'new_tests', 'new_positives']
    data = pd.DataFrame()
    for col in cols:
        assert col in raw_data.columns, f'Column "{col}" is missing from the loaded data'
    data = raw_data[cols]

    return data



