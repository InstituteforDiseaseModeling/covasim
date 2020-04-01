'''
Set the parameters for Covasim.
'''

import pandas as pd
from datetime import datetime


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
    pars['beta']         = 0.015 # Beta per symptomatic contact; absolute
    pars['asymp_factor'] = 0.8 # Multiply beta by this factor for asymptomatic cases
    pars['diag_factor']  = 0.0 # Multiply beta by this factor for diganosed cases -- baseline assumes complete isolation
    pars['cont_factor']  = 1.0 # Multiply beta by this factor for people who've been in contact with known positives  -- baseline assumes no isolation
    pars['contacts']     = 20 # Estimated number of contacts
    pars['beta_pop']     = {'H': 1.5,  'S': 1.5,   'W': 1.5,  'R': 0.5} # Per-population beta weights; relative
    pars['contacts_pop'] = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 20.0} # default flu-like weights # Number of contacts per person per day, estimated

    # Duration parameters: time for disease progression
    pars['dur'] = dict()
    pars['dur']['exp2inf']  = dict(dist='lognormal_int', par1=4, par2=1) # Duration from exposed to infectious
    pars['dur']['inf2sym']  = dict(dist='lognormal_int', par1=1, par2=1) # Duration from infectious to symptomatic
    pars['dur']['sym2sev']  = dict(dist='lognormal_int', par1=1, par2=1) # Duration from symptomatic to severe symptoms
    pars['dur']['sev2crit'] = dict(dist='lognormal_int', par1=1, par2=1) # Duration from severe symptoms to requiring ICU

    # Duration parameters: time for disease recovery
    pars['dur']['asym2rec'] = dict(dist='lognormal_int', par1=8, par2=2) # Duration for asymptomatics to recover
    pars['dur']['mild2rec'] = dict(dist='lognormal_int', par1=8, par2=2) # Duration from mild symptoms to recovered
    pars['dur']['sev2rec']  = dict(dist='lognormal_int', par1=11, par2=3) # Duration from severe symptoms to recovered - leads to mean total disease time of
    pars['dur']['crit2rec'] = dict(dist='lognormal_int', par1=17, par2=3) # Duration from critical symptoms to recovered

    # Duration parameters: time to die for critical cases
    pars['dur']['crit2die'] = dict(dist='lognormal_int', par1=21, par2=4) # Duration from critical symptoms to death

    # Severity parameters: probabilities of symptom progression
    pars['prog_by_age']         = True # Whether or not to use age-specific probabilities of prognosis (symptoms/severe symptoms/death)
    pars['default_symp_prob']   = 0.75 # If not using age-specific values: overall proportion of symptomatic cases
    pars['default_severe_prob'] = 0.12 # If not using age-specific values: proportion of symptomatic cases that become severe
    pars['default_crit_prob']   = 0.25 # If not using age-specific values: proportion of severe cases that become critical
    pars['default_death_prob']  = 0.5 # If not using age-specific values: proportion of critical cases that result in death
    pars['OR_no_treat']         = 2. # Odds ratio for how much more likely people are to die if no treatment available

    # Events and interventions
    pars['interventions'] = []  #: List of Intervention instances
    pars['interv_func'] = None # Custom intervention function

    # Health system parameters
    pars['n_beds'] = pars['n']  # Baseline assumption is that there's enough beds for the whole population (i.e., no constraints)

    return pars


def load_data(filename, datacols=None, **kwargs):
    '''
    Load data for comparing to the model output.

    Args:
        filename (str): the name of the file to load
        datacols (list): list of required column names
        kwargs (dict): passed to pd.read_excel()

    Returns:
        data (dataframe): pandas dataframe of the loaded data
    '''

    if datacols is None:
        datacols = ['day', 'date', 'new_tests', 'new_positives']

    # Load data
    raw_data = pd.read_excel(filename, **kwargs)

    # Confirm data integrity and simplify
    for col in datacols:
        assert col in raw_data.columns, f'Column "{col}" is missing from the loaded data'
    data = raw_data[datacols]

    return data



