'''
Set the parameters for Covasim.
'''

import numpy as np
import pandas as pd


__all__ = ['make_pars', 'set_contacts', 'get_prognoses', 'load_data']


def make_pars(set_prognoses=False, prog_by_age=True, use_layers=False, **kwargs):
    '''
    Set parameters for the simulation.

    Args:
        Set_prognoses (bool): whether or not to create prognoses (else, added when the population is created)
        prog_by_age (bool): whether or not to use age-based severity, mortality etc.
        use_layers (bool): whether or not to use household, school, etc. contact layers

    Returns:
        pars (dict): the parameters of the simulation
    '''
    pars = {}

    # Population parameters
    pars['pop_size']     = 20e3 # Number ultimately susceptible to CoV
    pars['pop_infected'] = 10 # Number of initial infections
    pars['pop_type']     = 'random' # What type of population data to use -- random (fastest), synthpops (best), hybrid (compromise), or clustered (not recommended)
    pars['location']     = None # What location to load data from -- default Seattle

    # Simulation parameters
    pars['start_day']  = '2020-03-01' # Start day of the simulation
    pars['n_days']     = 60 # Number of days of run, if end_day isn't used
    pars['rand_seed']  = 1 # Random seed, if None, don't reset
    pars['verbose']    = 1 # Whether or not to display information during the run -- options are 0 (silent), 1 (default), 2 (everything)

    # Rescaling parameters
    pars['pop_scale']         = 1   # Factor by which to scale the population -- e.g. 1000 with pop_size = 10e3 means a population of 10m
    pars['rescale']           = 0   # Enable dynamic rescaling of the population
    pars['rescale_threshold'] = 0.1 # Fraction susceptible population that will trigger rescaling if rescaling
    pars['rescale_factor']    = 2   # Factor by which we rescale the population

    # Basic disease transmission
    pars['beta']         = 0.015 # Beta per symptomatic contact; absolute
    pars['use_layers']   = use_layers # Whether or not to use different contact layers
    pars['contacts']     = None # The number of contacts per layer; set below
    pars['beta_layer']  = None # Transmissibility per layer
    pars['n_imports']    = 0 # Average daily number of imported cases (actual number is drawn from Poisson distribution)

    # Efficacy of protection measures
    pars['asymp_factor'] = 0.8 # Multiply beta by this factor for asymptomatic cases
    pars['diag_factor']  = 0.2 # Multiply beta by this factor for diganosed cases
    pars['quar_eff']     = None # Quarantine multiplier on transmissibility and susceptibility; set below
    pars['quar_period']  = 14  # Number of days to quarantine for -- TODO, should this be drawn from distribution, or fixed since it's policy?

    # Duration parameters: time for disease progression
    pars['dur'] = {}
    pars['dur']['exp2inf']  = {'dist':'lognormal_int', 'par1':4, 'par2':1} # Duration from exposed to infectious
    pars['dur']['inf2sym']  = {'dist':'lognormal_int', 'par1':1, 'par2':1} # Duration from infectious to symptomatic
    pars['dur']['sym2sev']  = {'dist':'lognormal_int', 'par1':1, 'par2':1} # Duration from symptomatic to severe symptoms
    pars['dur']['sev2crit'] = {'dist':'lognormal_int', 'par1':1, 'par2':1} # Duration from severe symptoms to requiring ICU

    # Duration parameters: time for disease recovery
    pars['dur']['asym2rec'] = {'dist':'lognormal_int', 'par1':8,  'par2':2} # Duration for asymptomatics to recover
    pars['dur']['mild2rec'] = {'dist':'lognormal_int', 'par1':8,  'par2':2} # Duration from mild symptoms to recovered
    pars['dur']['sev2rec']  = {'dist':'lognormal_int', 'par1':11, 'par2':3} # Duration from severe symptoms to recovered - leads to mean total disease time of
    pars['dur']['crit2rec'] = {'dist':'lognormal_int', 'par1':17, 'par2':3} # Duration from critical symptoms to recovered
    pars['dur']['crit2die'] = {'dist':'lognormal_int', 'par1':7,  'par2':3} # Duration from critical symptoms to death

    # Severity parameters: probabilities of symptom progression
    pars['OR_no_treat']     = 2.0  # Odds ratio for how much more likely people are to die if no treatment available
    pars['rel_symp_prob']   = 1.0  # Scale factor for proportion of symptomatic cases
    pars['rel_severe_prob'] = 1.0  # Scale factor for proportion of symptomatic cases that become severe
    pars['rel_crit_prob']   = 1.0  # Scale factor for proportion of severe cases that become critical
    pars['rel_death_prob']  = 1.0  # Scale factor for proportion of critical cases that result in death
    pars['prog_by_age']     = prog_by_age
    pars['prognoses']       = None # Populate this later

    # Events and interventions
    pars['interventions'] = []  #: List of Intervention instances
    pars['interv_func']   = None # Custom intervention function
    pars['timelimit']     = 3600 # Time limit for a simulation (seconds)
    pars['stopping_func'] = None # A function to call to stop the sim partway through

    # Health system parameters
    pars['n_beds'] = np.inf  # Baseline assumption is that there's no upper limit on the number of beds i.e. there's enough for everyone

    # Update with any supplied parameter values and generate things that need to be generated
    pars.update(kwargs)
    set_contacts(pars)
    if set_prognoses: # If not set here, gets set when the population is initialized
        pars['prognoses'] = get_prognoses(pars['prog_by_age']) # Default to age-specific prognoses

    return pars


def set_contacts(pars):
    '''
    Small helper function to set numbers of contacts and beta based on whether
    or not to use layers. Typically not called by the user.

    Args:
        pars (dict): the parameters dictionary
    '''
    if pars['use_layers']:
        if pars.get('contacts',   None) is None: pars['contacts']   = {'h': 4,   's': 20,  'w': 20,  'c': 10}   # Number of contacts per person per day, estimated
        if pars.get('beta_layer', None) is None: pars['beta_layer'] = {'h': 2.0, 's': 1.0, 'w': 1.0, 'c': 0.5}  # Per-population beta weights; relative
        if pars.get('quar_eff',   None) is None: pars['quar_eff']   = {'h': 0.5, 's': 0.0, 'w': 0.0, 'c': 0.05} # Multiply beta by this factor for people who know they've been in contact with a positive, even if they haven't been diagnosed yet
    else:
        if pars.get('contacts',   None) is None: pars['contacts']   = {'a': 20}  # Number of contacts per person per day -- 'a' for 'all'
        if pars.get('beta_layer', None) is None: pars['beta_layer'] = {'a': 1.0} # Per-population beta weights; relative
        if pars.get('quar_eff',   None) is None: pars['quar_eff']   = {'a': 0.3} # Multiply beta by this factor for people who know they've been in contact with a positive, even if they haven't been diagnosed yet
    return



def get_prognoses(by_age=True):
    '''
    Return the default parameter values for prognoses

    The prognosis probabilities are conditional given the previous disease state.

    Args:
        by_age (bool): whether or not to use age-specific values

    Returns:
        prog_pars (dict): the dictionary of prognosis probabilities

    '''

    max_age = 120 # For the sake of having a finite age cutoff

    if not by_age:
        prognoses = dict(
            age_cutoffs  = np.array([ max_age ]),
            symp_probs   = np.array([ 0.75 ]),
            severe_probs = np.array([ 0.2 ]),
            crit_probs   = np.array([ 0.08 ]),
            death_probs  = np.array([ 0.02 ]),
        )
    else:
        prognoses = dict(
            age_cutoffs  = np.array([10,      20,      30,      40,      50,      60,      70,      80,      max_age]), # Age cutoffs
            symp_probs   = np.array([0.50,    0.55,    0.60,    0.65,    0.70,    0.75,    0.80,    0.85,    0.90]),    # Overall probability of developing symptoms
            severe_probs = np.array([0.00100, 0.00100, 0.01100, 0.03400, 0.04300, 0.08200, 0.11800, 0.16600, 0.18400]), # Overall probability of developing severe symptoms (https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf)
            crit_probs   = np.array([0.00004, 0.00011, 0.00050, 0.00123, 0.00214, 0.00800, 0.02750, 0.06000, 0.10333]), # Overall probability of developing critical symptoms (derived from https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm)
            death_probs  = np.array([0.00002, 0.00006, 0.00030, 0.00080, 0.00150, 0.00600, 0.02200, 0.05100, 0.09300]), # Overall probability of dying (https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf)
        )

    prognoses['death_probs']  /= prognoses['crit_probs']   # Conditional probability of dying, given severe symptoms
    prognoses['crit_probs']   /= prognoses['severe_probs'] # Conditional probability of symptoms becoming critical, given severe
    prognoses['severe_probs'] /= prognoses['symp_probs']   # Conditional probability of symptoms becoming severe, given symptomatic

    return prognoses


def load_data(filename, columns=None, calculate=True, verbose=True, **kwargs):
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
                    if verbose:
                        print(f'  Automatically adding cumulative column {cum_col} from {col}')

    # Ensure required columns are present
    if 'date' not in data.columns:
        errormsg = f'Required column "date" not found; columns are {data.columns}'
        raise ValueError(errormsg)
    else:
        data['date'] = pd.to_datetime(data['date']).dt.date

    data.set_index('date', inplace=True)

    return data



