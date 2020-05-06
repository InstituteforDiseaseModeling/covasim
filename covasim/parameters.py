'''
Set the parameters for Covasim.
'''

import numpy as np


__all__ = ['make_pars', 'reset_layer_pars', 'get_prognoses']


def make_pars(set_prognoses=False, prog_by_age=True, **kwargs):
    '''
    Set parameters for the simulation.

    Args:
        Set_prognoses (bool): whether or not to create prognoses (else, added when the population is created)
        prog_by_age (bool): whether or not to use age-based severity, mortality etc.

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
    pars['end_day']    = None # End day of the simulation
    pars['n_days']     = 60 # Number of days to run, if end_day isn't specified
    pars['rand_seed']  = 1 # Random seed, if None, don't reset
    pars['verbose']    = 1 # Whether or not to display information during the run -- options are 0 (silent), 1 (default), 2 (everything)

    # Rescaling parameters
    pars['pop_scale']         = 1    # Factor by which to scale the population -- e.g. pop_scale=10 with pop_size=100e3 means a population of 1 million
    pars['rescale']           = 0    # Enable dynamic rescaling of the population -- starts with pop_scale=1 and scales up dynamically as the epidemic grows
    pars['rescale_threshold'] = 0.05 # Fraction susceptible population that will trigger rescaling if rescaling
    pars['rescale_factor']    = 2    # Factor by which we rescale the population

    # Basic disease transmission
    pars['beta']        = 0.016 # Beta per symptomatic contact; absolute value, calibrated
    pars['contacts']    = None # The number of contacts per layer; set below
    pars['dynam_layer'] = None # Which layers are dynamic; set below
    pars['beta_layer']  = None # Transmissibility per layer; set below
    pars['n_imports']   = 0 # Average daily number of imported cases (actual number is drawn from Poisson distribution)
    pars['beta_dist']   = {'dist':'lognormal','par1':0.84, 'par2':0.3} # Distribution to draw individual level transmissibility; see https://wellcomeopenresearch.org/articles/5-67
    pars['viral_dist']  = {'frac_time':0.3, 'load_ratio':2, 'high_cap':4} # The time varying viral load (transmissibility); estimated from Lescure 2020, Lancet, https://doi.org/10.1016/S1473-3099(20)30200-0

    # Efficacy of protection measures
    pars['asymp_factor'] = 1.0 # Multiply beta by this factor for asymptomatic cases; no statistically significant difference in transmissibility: https://www.sciencedirect.com/science/article/pii/S1201971220302502
    pars['diag_factor']  = 0.2 # Multiply beta by this factor for diganosed cases; based on intervention strength
    pars['quar_eff']     = None # Quarantine multiplier on transmissibility and susceptibility; set below
    pars['quar_period']  = 14  # Number of days to quarantine for; assumption based on standard policies

    # Duration parameters: time for disease progression
    pars['dur'] = {}
    pars['dur']['exp2inf']  = {'dist':'lognormal_int', 'par1':4.6, 'par2':4.8} # Duration from exposed to infectious; see Linton et al., https://doi.org/10.3390/jcm9020538
    pars['dur']['inf2sym']  = {'dist':'lognormal_int', 'par1':1.0, 'par2':0.9} # Duration from infectious to symptomatic; see Linton et al., https://doi.org/10.3390/jcm9020538
    pars['dur']['sym2sev']  = {'dist':'lognormal_int', 'par1':6.6, 'par2':4.9} # Duration from symptomatic to severe symptoms; see Linton et al., https://doi.org/10.3390/jcm9020538
    pars['dur']['sev2crit'] = {'dist':'lognormal_int', 'par1':3.0, 'par2':7.4} # Duration from severe symptoms to requiring ICU; see Wang et al., https://jamanetwork.com/journals/jama/fullarticle/2761044

    # Duration parameters: time for disease recovery
    pars['dur']['asym2rec'] = {'dist':'lognormal_int', 'par1':8.0,  'par2':2.0} # Duration for asymptomatics to recover; see Wölfel et al., https://www.nature.com/articles/s41586-020-2196-x
    pars['dur']['mild2rec'] = {'dist':'lognormal_int', 'par1':8.0,  'par2':2.0} # Duration from mild symptoms to recovered; see Wölfel et al., https://www.nature.com/articles/s41586-020-2196-x
    pars['dur']['sev2rec']  = {'dist':'lognormal_int', 'par1':14.0, 'par2':2.4} # Duration from severe symptoms to recovered, 22.6 days total; see Verity et al., https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf
    pars['dur']['crit2rec'] = {'dist':'lognormal_int', 'par1':14.0, 'par2':2.4} # Duration from critical symptoms to recovered, 22.6 days total; see Verity et al., https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf
    pars['dur']['crit2die'] = {'dist':'lognormal_int', 'par1':6.2,  'par2':1.7} # Duration from critical symptoms to death, 17.8 days total; see Verity et al., https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf

    # Severity parameters: probabilities of symptom progression
    pars['OR_no_treat']     = 2.0  # Odds ratio for how much more likely people are to die if no treatment available
    pars['rel_symp_prob']   = 1.0  # Scale factor for proportion of symptomatic cases
    pars['rel_severe_prob'] = 1.0  # Scale factor for proportion of symptomatic cases that become severe
    pars['rel_crit_prob']   = 1.0  # Scale factor for proportion of severe cases that become critical
    pars['rel_death_prob']  = 1.0  # Scale factor for proportion of critical cases that result in death
    pars['prog_by_age']     = prog_by_age # Whether to set disease progression based on the person's age
    pars['prognoses']       = None # Populate this later

    # Events and interventions
    pars['interventions'] = []   # List of Intervention instances
    pars['interv_func']   = None # Custom intervention function
    pars['timelimit']     = 3600 # Time limit for a simulation (seconds)
    pars['stopping_func'] = None # A function to call to stop the sim partway through

    # Health system parameters
    pars['n_beds'] = None  # Baseline assumption is that there's no upper limit on the number of beds i.e. there's enough for everyone

    # Update with any supplied parameter values and generate things that need to be generated
    pars.update(kwargs)
    reset_layer_pars(pars)
    if set_prognoses: # If not set here, gets set when the population is initialized
        pars['prognoses'] = get_prognoses(pars['prog_by_age']) # Default to age-specific prognoses

    return pars


def reset_layer_pars(pars, layer_keys=None, force=False):
    '''
    Small helper function to set numbers of contacts and beta based on whether
    or not to use layers.

    Args:
        pars (dict): the parameters dictionary
        pop_keys (list): the known keys of the population, if available
        force (bool): reset the pars even if they already exist
    '''
    d_contacts    = 20  # Default number of contacts
    d_dynam_layer = 0   # Do not use dynamic layers by default
    d_beta_layer  = 1.0 # No change in beta
    d_quar_eff    = 0.3 # Assumed quarantine effectiveness

    if layer_keys is not None: # Create based on known population keys
        pars['contacts']    = {lkey:d_contacts    for lkey in layer_keys}
        pars['dynam_layer'] = {lkey:d_dynam_layer for lkey in layer_keys}
        pars['beta_layer']  = {lkey:d_beta_layer  for lkey in layer_keys}
        pars['quar_eff']    = {lkey:d_quar_eff    for lkey in layer_keys}
    else: # Guess based on population type
        if pars['pop_type'] == 'random':
            if pars.get('contacts',    None) is None or force: pars['contacts']    = {'a': d_contacts}    # Number of contacts per person per day -- 'a' for 'all'
            if pars.get('dynam_layer', None) is None or force: pars['dynam_layer'] = {'a': d_dynam_layer} # Which layers are dynamic
            if pars.get('beta_layer',  None) is None or force: pars['beta_layer']  = {'a': d_beta_layer}  # Per-population beta weights; relative
            if pars.get('quar_eff',    None) is None or force: pars['quar_eff']    = {'a': d_quar_eff}    # Multiply beta by this factor for people who know they've been in contact with a positive, even if they haven't been diagnosed yet
        else:
            if pars.get('contacts',    None) is None or force: pars['contacts']    = dict(h=2.7, s=20,  w=8,  c=20)   # Number of contacts per person per day, estimated
            if pars.get('dynam_layer', None) is None or force: pars['dynam_layer'] = dict(h=0,   s=0,   w=0,   c=0)    # Which layers are dynamic -- none by defaul
            if pars.get('beta_layer',  None) is None or force: pars['beta_layer']  = dict(h=7.0, s=0.7, w=1.4, c=0.14)  # Per-population beta weights; relative
            if pars.get('quar_eff',    None) is None or force: pars['quar_eff']    = dict(h=0.5, s=0.0, w=0.0, c=0.05) # Multiply beta by this factor
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
            age_cutoffs   = np.array([10,      20,      30,      40,      50,      60,      70,      80,      max_age]), # Age cutoffs (upper limits)
            sus_ORs       = np.array([0.34,    0.67,    1.00,    1.00,    1.00,    1.00,    1.24,    1.47,    1.47]),    # Odds ratios for relative susceptibility -- from https://science.sciencemag.org/content/early/2020/05/04/science.abb8001; 10-20 and 60-70 bins are the average across the ORs
            trans_ORs     = np.array([1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00]),    # Odds ratios for relative transmissibility -- no evidence of differences
            comorbidities = np.array([1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00]),    # Comorbidities by age -- set to 1 by default since already included in disease progression rates
            symp_probs    = np.array([0.50,    0.55,    0.60,    0.65,    0.70,    0.75,    0.80,    0.85,    0.90]),    # Overall probability of developing symptoms (based on https://www.medrxiv.org/content/10.1101/2020.03.24.20043018v1.full.pdf, scaled for overall symptomaticity)
            severe_probs  = np.array([0.00050, 0.00165, 0.00720, 0.02080, 0.03430, 0.07650, 0.13280, 0.20655, 0.24570]), # Overall probability of developing severe symptoms (derived from Table 1 of https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf)
            crit_probs    = np.array([0.00003, 0.00008, 0.00036, 0.00104, 0.00216, 0.00933, 0.03639, 0.08923, 0.17420]), # Overall probability of developing critical symptoms (derived from Table 1 of https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf)
            death_probs   = np.array([0.00002, 0.00006, 0.00030, 0.00080, 0.00150, 0.00600, 0.02200, 0.05100, 0.09300]), # Overall probability of dying (https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf)
        )

    prognoses['death_probs']  /= prognoses['crit_probs']   # Conditional probability of dying, given critical symptoms
    prognoses['crit_probs']   /= prognoses['severe_probs'] # Conditional probability of symptoms becoming critical, given severe
    prognoses['severe_probs'] /= prognoses['symp_probs']   # Conditional probability of symptoms becoming severe, given symptomatic

    return prognoses
