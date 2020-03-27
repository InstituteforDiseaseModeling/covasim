'''
Set the parameters for Covasim.
'''

import pylab as pl
import pandas as pd
from datetime import datetime
import numba as nb
import sciris as sc


__all__ = ['make_pars', 'set_person_attrs', 'set_cfr', 'set_severity', 'load_data']


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
    pars['diag_factor']    = 0. # Multiply beta by this factor for diganosed cases -- baseline assumes no isolation
    pars['cont_factor']    = 1.0 # Multiply beta by this factor for people who've been in contact with known positives  -- baseline assumes no isolation
    pars['contacts']       = 20
    pars['beta_pop']       = {'H': 1.5,  'S': 1.5,   'W': 1.5,  'R': 0.5} # Per-population beta weights; relative
    pars['contacts_pop']   = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 20.0} # default flu-like weights # Number of contacts per person per day, estimated

    # Disease progression
    pars['serial']         = 4.0 # Serial interval: days after exposure before a person can infect others (see e.g. https://www.ncbi.nlm.nih.gov/pubmed/32145466)
    pars['serial_std']     = 1.0 # Standard deviation of the serial interval
    pars['incub']          = 5.0 # Incubation period: days until an exposed person develops symptoms
    pars['incub_std']      = 1.0 # Standard deviation of the incubation period
    pars['dur']            = 8 # Using Mike's Snohomish number
    pars['dur_std']        = 2 # Variance in duration

    # Mortality and severity
    pars['timetodie']           = 21 # Days until death
    pars['timetodie_std']       = 2 # STD
    pars['severity_by_age']     = True # Whether or not to use age-specific probabilities of developing severe infection
    pars['default_severity']    = 0.2 # If not using age-specific values: overall proportion of severe cases
    pars['asymp_prop']          = 0.2 # Proportion of asymptomatic cases

    # Events and interventions
    pars['interventions'] = []  #: List of Intervention instances
    pars['interv_func'] = None # Custom intervention function

    return pars


@nb.njit()
def _get_norm_age(min_age, max_age, age_mean, age_std):
    norm_age = pl.normal(age_mean, age_std)
    age = pl.minimum(pl.maximum(norm_age, min_age), max_age)
    return age


def set_person_attrs(min_age=0, max_age=99, age_mean=40, age_std=15, default_severity=None, severity_by_age=True, use_data=True):
    '''
    Set the attributes for an individual, including:
        * age
        * sex
        * severity (i.e., how likely they are to develop severe symptoms -- based on age)
    '''
    sex = pl.randint(2) # Define female (0) or male (1) -- evenly distributed
    age = _get_norm_age(min_age, max_age, age_mean, age_std)

    # Get the probability of developing severe symptoms for a person of this age
    severity = set_severity(age=age, default_severity=default_severity, severity_by_age=severity_by_age, severity_fn=severity_fn, max_age=max_age)

    return age, sex, severity


def set_prognosis(age=None, default_severity=0.2, by_age=True):
    '''
    Determine the prognosis of an infected person: probability of developing severe symptoms and dying, based on their age
    '''

    # Probabilities of death after onset of severe symptoms
    age_cutoffs  = [10,      20,      30,      40,      50,      60,      70,      80,      100]
    symp_probs   = [0.00000, 0.00000, 0.01100, 0.03400, 0.04300, 0.08200, 0.11800, 0.16600, 0.18400]
    severe_probs = [0.00000, 0.00000, 0.01100, 0.03400, 0.04300, 0.08200, 0.11800, 0.16600, 0.18400]
    death_probs  = [0.00002, 0.00006, 0.00030, 0.00080, 0.00150, 0.00600, 0.02200, 0.05100, 0.09300]

    fr_if_severe = [d/s if s>0 and d/s>0 else 0 for (d,s) in zip(death_props,severe_props)] # Fatality rate among those severe symptoms who die, by age

    # Process different options for age
    # Not supplied, use default
    if age is None or not severity_by_age:
        severity = default_severity

    # Single number
    elif sc.isnumber(age):

        # Define the age-dependent probabilities of developing severe infection
        max_age_death_severe = death_probs[-1]
        max_age_severity     = severe_probs[-1]

        # Figure out which probability applies to a person of the specified age
        severityind = next((ind for ind, val in enumerate([True if age < cutoff else False for cutoff in age_cutoffs]) if val), max_age_severity)
        severity    = severe_probs[severityind] # Probability of developing severe symptoms
        deathind    = next((ind for ind, val in enumerate([True if age < cutoff else False for cutoff in age_cutoffs]) if val), max_age_death_severe)
        death       = death_severe[deathind] # Probability of dying after developing severe symptoms

    # Listlike
    elif sc.checktype(age, 'listlike'):
        severity = []
        for a in age: severity.append(set_severity(age=a, default_severity=default_severityfr, severitydict=severitydict, severity_by_age=severity_by_age))

    else:
        raise TypeError(f"set_severity accepts a single age or list/aray of ages, not type {type(age)}")

    return severity


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



