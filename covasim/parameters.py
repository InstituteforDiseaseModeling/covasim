'''
Set the parameters for Covasim.
'''

import numpy as np
import pandas as pd
import sciris as sc
import datetime as dt


__all__ = ['make_pars', 'get_default_prognoses', 'load_data']


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
    pars['start_day']  = dt.datetime(2020, 3, 1) # Start day of the simulation
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
    pars['contacts']     = {'h': 4,   's': 10,  'w': 10,  'c': 20} # Number of contacts per person per day, estimated
    pars['beta_layers']  = {'h': 1.7, 's': 0.8, 'w': 0.8, 'c': 0.3} # Per-population beta weights; relative

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
    pars['dur']['crit2die'] = dict(dist='lognormal_int', par1=21, par2=4) # Duration from critical symptoms to death

    # Severity parameters: probabilities of symptom progression
    pars['prog_by_age']     = True # Whether or not to use age-specific probabilities of prognosis (symptoms/severe symptoms/death)
    pars['rel_symp_prob']   = 1.0  # If not using age-specific values: relative proportion of symptomatic cases
    pars['rel_severe_prob'] = 1.0  # If not using age-specific values: relative proportion of symptomatic cases that become severe
    pars['rel_crit_prob']   = 1.0  # If not using age-specific values: relative proportion of severe cases that become critical
    pars['rel_death_prob']  = 1.0  # If not using age-specific values: relative proportion of critical cases that result in death
    pars['OR_no_treat']     = 2.0  # Odds ratio for how much more likely people are to die if no treatment available

    # Events and interventions
    pars['interventions'] = []  #: List of Intervention instances
    pars['interv_func'] = None # Custom intervention function

    # Health system parameters
    pars['n_beds'] = pars['n']  # Baseline assumption is that there's enough beds for the whole population (i.e., no constraints)

    return pars



def get_default_prognoses(by_age=True):
    '''
    Return the default parameter values for prognoses

    Args:
        by_age (bool): whether or not to use age-specific values

    Returns:
        prog_pars (dict): the dictionary of prognosis probabilities

    '''
    if not by_age:
        prog_pars = sc.objdict(
            symp_prob   = 0.75,
            severe_prob = 0.12,
            crit_prob   = 0.25,
            death_prob  = 0.50,
        )
    else:
        prog_pars = sc.objdict(
            age_cutoffs  = np.array([10,      20,      30,      40,      50,      60,      70,      80,      120]),     # Age cutoffs
            symp_probs   = np.array([0.50,    0.55,    0.60,    0.65,    0.70,    0.75,    0.80,    0.85,    0.90]),    # Overall probability of developing symptoms
            severe_probs = np.array([0.00100, 0.00100, 0.01100, 0.03400, 0.04300, 0.08200, 0.11800, 0.16600, 0.18400]), # Overall probability of developing severe symptoms (https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf)
            crit_probs   = np.array([0.00004, 0.00011, 0.00050, 0.00123, 0.00214, 0.00800, 0.02750, 0.06000, 0.10333]), # Overall probability of developing critical symptoms (derived from https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm)
            death_probs  = np.array([0.00002, 0.00006, 0.00030, 0.00080, 0.00150, 0.00600, 0.02200, 0.05100, 0.09300]), # Overall probability of dying (https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf)
        )
    return prog_pars




def load_data(filename, columns=None, calculate=True, **kwargs):
    '''
    Load data for comparing to the model output.

    Args:
        filename (str): the name of the file to load (either Excel or CSV)
        columns (list): list of column names (otherwise, load all)
        calculate (bool): whether or not to calculate cumulative values from daily counts
        kwargs (dict): passed to pd.read_excel()

    Returns:
        data (dataframe): pandas dataframe of the loaded data
    '''

    # Load data
    if filename.lower().endswith('csv'):
        raw_data = pd.read_csv(filename, **kwargs)
    elif filename.lower().endswith('xlsx'):
        raw_data = pd.read_excel(filename, **kwargs)
    else:
        errormsg = f'Currently loading is only supported from .csv and .xlsx files, not {filename}'
        raise NotImplementedError(errormsg)

    # Confirm data integrity and simplify
    if columns is not None:
        for col in columns:
            if col not in raw_data.columns:
                errormsg = f'Column "{col}" is missing from the loaded data'
                raise ValueError(errormsg)
        data = raw_data[columns]
    else:
        data = raw_data

    # Calculate any cumulative columns that are missing
    if calculate:
        columns = data.columns
        for col in columns:
            if col.startswith('new'):
                cum_col = col.replace('new_', 'cum_')
                if cum_col not in columns:
                    data[cum_col] = np.cumsum(data[col])

    # Ensure required columns are present
    if 'date' not in data.columns:
        errormsg = f'Required column "date" not found; columns are {data.columns}'
        raise ValueError(errormsg)
    else:
        data['date'] = pd.to_datetime(data['date'])

    return data



