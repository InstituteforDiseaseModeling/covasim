'''
Set the parameters for Covasim.
'''

import pylab as pl
import pandas as pd
from datetime import datetime
import numba as nb
import sciris as sc


__all__ = ['make_pars', 'set_person_attrs', 'set_prognosis', 'load_data']


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
    pars['prog_by_age']         = True # Whether or not to use age-specific probabilities of prognosis (symptoms/severe symptoms/death)
    pars['default_symp_prob']   = 0.7 # If not using age-specific values: overall proportion of symptomatic cases
    pars['default_severe_prob'] = 0.3 # If not using age-specific values: proportion of symptomatic cases that become severe (default 0.2 total)
    pars['default_death_prob']  = 0.07 # If not using age-specific values: proportion of severe cases that result in death (default 0.02 CFR)

    # Events and interventions
    pars['interventions'] = []  #: List of Intervention instances
    pars['interv_func'] = None # Custom intervention function

    return pars


@nb.njit()
def _get_norm_age(min_age, max_age, age_mean, age_std):
    norm_age = pl.normal(age_mean, age_std)
    age = pl.minimum(pl.maximum(norm_age, min_age), max_age)
    return age


def set_person_attrs(min_age=0, max_age=99, age_mean=40, age_std=15, default_symp_prob=None, default_severe_prob=None,
                     default_death_prob=None, by_age=True, use_data=True):
    '''
    Set the attributes for an individual, including:
        * age
        * sex
        * prognosis (i.e., how likely they are to develop symptoms/develop severe symptoms/die, based on age)
    '''
    sex = pl.randint(2) # Define female (0) or male (1) -- evenly distributed
    age = _get_norm_age(min_age, max_age, age_mean, age_std)

    # Get the prognosis for a person of this age
    symp_prob, severe_prob, death_prob = set_prognosis(age=age, default_symp_prob=default_symp_prob, default_severe_prob=default_severe_prob, default_death_prob=default_death_prob, by_age=by_age)

    return age, sex, symp_prob, severe_prob, death_prob


def set_prognosis(age=None, default_symp_prob=0.7, default_severe_prob=0.2, default_death_prob=0.02, by_age=True):
    '''
    Determine the prognosis of an infected person: probability of being aymptomatic, or if symptoms develop, probability
    of developing severe symptoms and dying, based on their age
    '''
    # Overall probabilities of symptoms, severe symptoms, and death
    age_cutoffs  = [10,      20,      30,      40,      50,      60,      70,      80,      100]
    symp_probs   = [0.50,    0.55,    0.65,    0.70,    0.75,    0.80,    0.85,    0.90,    0.95]    # Overall probability of developing symptoms
    severe_probs = [0.00000, 0.00000, 0.01100, 0.03400, 0.04300, 0.08200, 0.11800, 0.16600, 0.18400] # Overall probability of developing severe symptoms (https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf)
    death_probs  = [0.00002, 0.00006, 0.00030, 0.00080, 0.00150, 0.00600, 0.02200, 0.05100, 0.09300] # Overall probability of dying (https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf)

    # Conditional probabilities of severe symptoms (given symptomatic) and death (given severe symptoms)
    severe_if_sym   = [sev/sym if sym>0 and sev/sym>0 else 0 for (sev,sym) in zip(severe_probs,symp_probs)]   # Conditional probabilty of developing severe symptoms, given symptomatic
    death_if_severe = [d/s if s>0 and d/s>0 else 0 for (d,s) in zip(death_probs,severe_probs)]                # Conditional probabilty of dying, given severe symptoms

    # Process different options for age
    # Not supplied, use default
    if age is None or not by_age:
        symp_prob, severe_prob, death_prob = default_symp_prob, default_severe_prob, default_death_prob

    # Single number
    elif sc.isnumber(age):

        # Figure out which probability applies to a person of the specified age
        ind = next((ind for ind, val in enumerate([True if age < cutoff else False for cutoff in age_cutoffs]) if val), -1)
        symp_prob    = symp_probs[ind]    # Probability of developing symptoms
        severe_prob = severe_if_sym[ind] # Probability of developing severe symptoms
        death_prob  = death_if_severe[ind] # Probability of dying after developing severe symptoms

    # Listlike
    elif sc.checktype(age, 'listlike'):
        symp_prob, severe_prob, death_prob  = [],[],[]
        for a in age:
            this_symp_prob, this_severe_prob, this_death_prob = set_prognosis(age=age, default_symp_prob=default_symp_prob, default_severe_prob=default_severe_prob, default_death_prob=default_death_prob, by_age=by_age)
            symp_prob.append(this_symp_prob)
            severe_prob.append(this_severe_prob)
            death_prob.append(this_death_prob)

    else:
        raise TypeError(f"set_prognosis accepts a single age or list/aray of ages, not type {type(age)}")

    return symp_prob, severe_prob, death_prob


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



