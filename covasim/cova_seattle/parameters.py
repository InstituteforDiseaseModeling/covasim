'''
Set the parameters for COVID-ABM.
'''

import pylab as pl


__all__ = ['make_pars', 'get_age_sex']


def make_pars():
    ''' Set parameters for the simulation '''
    pars = {}

    # Simulation parameters
    pars['scale']       = 25 # Factor by which to scale results ## 100

    pars['n']           = int(0.1 * 0.4 * 3e6 // pars['scale']) # Number ultimately susceptible to CoV
    pars['n_infected']  = 100 // pars['scale'] # Asked for 1000 in Seattle's population # 550
    pars['day_0']       = 53 #datetime(2020, 2, 10) # Start day of the epidemic 3/5
    pars['n_days']      = 45 # 75 #(datetime(2020, 4, 28)-pars['day_0']).days # How many days to simulate Apr/30 # 54
    pars['seed']        = 1 # Random seed, if None, don't reset
    pars['verbose']     = 1 # Whether or not to display information during the run -- options are 0 (silent), 1 (default), 2 (everything)
    pars['usepopdata']  = 0 # Whether or not to load actual population data
    pars['cfr_by_age']  = 1 # Whether or not to use age-specific case fatality
    pars['default_cfr'] = 0.02 # Default overall case fatality rate if not using age-specific values

    # Epidemic parameters
    pars['r_contact']      = 2.0/(8*10) # Updated to match Mike's distributions
    pars['contacts']       = 10 # Number of contacts per person per day, estimated
    pars['incub']          = 4.0 # Using Mike's Snohomish number
    pars['incub_std']      = 1.0 # Standard deviation of the serial interval, estimated
    pars['dur']            = 8 # Using Mike's Snohomish number
    pars['dur_std']        = 2 # Variance in duration
    pars['sensitivity']    = 1.0 # Probability of a true positive, estimated
    pars['symptomatic']    = 5 # Increased probability of testing someone symptomatic, estimated
    pars['timetodie']      = 22 # Days until death
    pars['timetodie_std']  = 2 # STD

    # Events
    pars['quarantine']       = -1  # Day on which quarantine took effect
    pars['unquarantine']     = -1  # Day on which unquarantine took effect
    pars['quarantine_eff']   = 1.00 # Change in transmissibility due to quarantine, estimated

    return pars


def get_age_sex(min_age=0, max_age=99, age_mean=40, age_std=15, cfr_by_age=True, use_data=True):
    '''
    Define age-sex distributions.
    '''
    if use_data:
        try:
            import synthpops as sp
        except ImportError as E:
            raise ImportError(f'Could not load synthpops; set sim["usepopdata"] = False or install ({str(E)})')
        age, sex = sp.get_seattle_age_sex()
    else:
        sex = pl.randint(2) # Define female (0) or male (1) -- evenly distributed
        age = pl.normal(age_mean, age_std) # Define age distribution for the crew and guests
        age = pl.median([min_age, age, max_age]) # Normalize

    # Get case fatality rate for a person of this age
    age_for_cfr = age if cfr_by_age else None # Whether or not to use age-specific values
    cfr = get_cfr(age=age_for_cfr)

    return age, sex, cfr


def get_cfr(age=None, default_cfr=0.02, cfrdict=None):
    '''
    Get age-dependent case-fatality rates
    '''
    # Check inputs and assign default CFR if age not supplied
    if age is None:
        print(f'No age given, using default case fatality rate of {default_cfr}...')
        cfr = default_cfr

    else:
        # Define age-dependent case fatality rates if not given
        if cfrdict is None:
            cfrdict = {'cutoffs': [9, 19, 29, 39, 49, 59, 69, 79, 99],
                       'values': [0.0001, 0.0002, 0.0009, 0.0018, 0.004, 0.013, 0.046, 0.098, 0.18]} # Table 1 of https://www.medrxiv.org/content/10.1101/2020.03.04.20031104v1.full.pdf

        # Figure out which CFR applies to a person of the specified age
        cfrind = next((ind for ind, val in enumerate([True if age<cutoff else False for cutoff in cfrdict['cutoffs']]) if val))
        cfr = cfrdict['values'][cfrind]

    return cfr

