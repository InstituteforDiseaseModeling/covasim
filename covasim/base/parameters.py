'''
Set the parameters for COVID-ABM.
'''

import pylab as pl
import pandas as pd
from datetime import datetime
import numba as nb
import sciris as sc


__all__ = ['make_pars', 'set_person_attributes', 'set_cfr', 'set_severity', 'load_data']


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

    # Disease transmission
    pars['beta']           = 0.015 # Beta per symptomatic contact; absolute
    pars['asym_prop']      = 0.17 # Proportion of asymptomatic cases - estimate based on https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2020.25.10.2000180, #TODO: look for better estimates
    pars['asym_factor']    = 0.8 # Multiply beta by this factor for asymptomatic cases
    pars['diag_factor']    = 1.0 # Multiply beta by this factor for diganosed cases -- baseline assumes no isolation
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

    # Testing
    pars['daily_tests']    = [] # If there's no testing data, optionally define a list of daily tests here. Remember this gets scaled by pars['scale']. To say e.g. 1% of the population is tested, use [0.01*pars['n']]*pars['n_days']
    pars['sensitivity']    = 1.0 # Probability of a true positive, estimated
    pars['sympt_test']     = 100.0 # Multiply testing probability by this factor for symptomatic cases
    pars['trace_test']     = 1.0 # Multiply testing probability by this factor for contacts of known positives -- baseline assumes no contact tracing

    # Mortality and severity
    pars['timetodie']           = 21 # Days until death
    pars['timetodie_std']       = 2 # STD
    pars['cfr_by_age']          = 0 # Whether or not to use age-specific case fatality
    pars['default_cfr']         = 0.016 # Default overall case fatality rate if not using age-specific values
    pars['severity_by_age']     = 0 # Whether or not to use age-specific case fatality
    pars['default_severity']    = 0.3 # Default overall case fatality rate if not using age-specific values

    # Events and interventions
    pars['interv_days'] = [] # Day on which interventions started/stopped, e.g. [30, 44]
    pars['interv_effs'] = [] # Change in transmissibility, e.g. [0.1, 10]
    pars['interv_func'] = None # Custom intervention function

    return pars


@nb.njit()
def _get_norm_age(min_age, max_age, age_mean, age_std):
    norm_age = pl.normal(age_mean, age_std)
    age = pl.minimum(pl.maximum(norm_age, min_age), max_age)
    return age


def set_person_attributes(min_age=0, max_age=99, age_mean=40, age_std=15, default_cfr=None, default_severity=None,
                          severity_fn=None, severity_by_age=True, cfr_by_age=True, use_data=True):
    '''
    Set the attributes for an individual, including:
        * age
        * sex
        * severity (i.e., how likely they are to develop symptoms -- based on age)
        * case-fatality rate (i.e., how likely they are to die -- based on age)
    '''
    sex = pl.randint(2) # Define female (0) or male (1) -- evenly distributed
    age = _get_norm_age(min_age, max_age, age_mean, age_std)

    # Get case fatality rate for a person of this age
    cfr = set_cfr(age=age, default_cfr=default_cfr, cfr_by_age=cfr_by_age)

    # Get symptom severity for a person of this age
    severity = set_severity(age=age, default_severity=default_severity, severity_by_age=severity_by_age, severity_fn=severity_fn, max_age=max_age)

    return age, sex, cfr, severity


def set_cfr(age=None, default_cfr=0.02, cfrdict=None, cfr_by_age=True):
    '''
    Set age-dependent case-fatality rates
    '''

    # Process different options for age
    # Not supplied, use default
    if age is None or not cfr_by_age:
        cfr = default_cfr

    # Single number
    elif sc.isnumber(age):

        # Define age-dependent case fatality rates if not given
        if cfrdict is None:
            cfrdict = {'cutoffs': [10, 20, 30, 40, 50, 60, 70, 80, 100],  # Age cutoffs
                       'values': [0.0001, 0.0002, 0.0009, 0.0018, 0.004, 0.013, 0.046, 0.098,
                                  0.18]}  # Table 1 of https://www.medrxiv.org/content/10.1101/2020.03.04.20031104v1.full.pdf
        max_age_cfr = cfrdict['values'][-1]  # For people older than the oldest

        # Figure out which CFR applies to a person of the specified age
        cfrind = next((ind for ind, val in enumerate([True if age < cutoff else False for cutoff in cfrdict['cutoffs']]) if val), max_age_cfr)
        cfr = cfrdict['values'][cfrind]

    # Listlike
    elif sc.checktype(age, 'listlike'):
        cfr = []
        for a in age: cfr.append(set_cfr(age=a, default_cfr=default_cfr, cfrdict=cfrdict, cfr_by_age=cfr_by_age))

    else:
        raise TypeError(f"set_cfr accepts a single age or list/aray of ages, not type {type(age)}")

    return cfr


def set_severity(age=None, default_severity=0.3, severity_by_age=True, severity_fn=None, max_age=100):
    '''
    Set symptom severity
    Desired features:
    1. Overall severity distribution:
           a. ~30% of cases should be asymptomatic
           b. ~50% of cases should be mild
           c. ~15% of cases should be severe
           d. ~5% of cases should be critical
    2. Severity by age distribution: older people should be more likely to become severe/critical

    Sources:
        https://jamanetwork.com/journals/jama/fullarticle/2762130 -- distribution in China
        https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30566-3/fulltext -- "increasing odds of in-hospital death associated with older age (odds ratio 1·10)"
        https://www.medrxiv.org/content/10.1101/2020.03.16.20037259v1.full.pdf -- asymptomaticity in different age groups in China
        https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf -- 2/3 symptomatic

    Implemented approach: use a simple linear relationship btwn age and severity index
    Alternative approach: (for future consideration) use a beta distribution (often used as a prior over binomial probs)
    '''

    if severity_fn is None: severity_fn = 'linear' # Default to linear

    # Process different options for age
    # Not supplied, use default
    if age is None or not severity_by_age:
        severity = default_severity

    # Single number
    elif sc.isnumber(age):
        if severity_fn == 'linear':
            severity = 0.5+0.5*age/max_age # Simple function designed to give desired features
        else:
            raise NotImplementedError('Only accepting linear severity function at the moment.')

    # Listlike
    elif sc.checktype(age, 'listlike'):
        severity = []
        for a in age: severity.append(set_severity(age=a, default_severity=default_severity, severity_by_age=severity_by_age, severity_fn=severity_fn, max_age=max_age))
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



