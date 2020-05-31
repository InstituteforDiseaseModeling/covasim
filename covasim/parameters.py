'''
Set the parameters for Covasim.
'''

import numpy as np
import sciris as sc

__all__ = ['make_pars', 'reset_layer_pars', 'get_prognoses']


def make_pars(set_prognoses=False, prog_by_age=True, **kwargs):
    '''
    Create the parameters for the simulation. Typically, this function is used
    internally rather than called by the user; e.g. typical use would be to do
    sim = cv.Sim() and then inspect sim.pars, rather than calling this function
    directly.

    Args:
        set_prognoses (bool): whether or not to create prognoses (else, added when the population is created)
        prog_by_age (bool): whether or not to use age-based severity, mortality etc.
        kwargs (dict): any additional kwargs are interpreted as parameter names

    Returns:
        pars (dict): the parameters of the simulation
    '''
    pars = {}

    # Population parameters
    pars['pop_size']     = 20e3     # Number of agents, i.e., people susceptible to SARS-CoV-2
    pars['pop_infected'] = 10       # Number of initial infections
    pars['pop_type']     = 'random' # What type of population data to use -- random (fastest), synthpops (best), hybrid (compromise), or clustered (not recommended)
    pars['location']     = None     # What location to load data from -- default Seattle

    # Simulation parameters
    pars['start_day']  = '2020-03-01' # Start day of the simulation
    pars['end_day']    = None         # End day of the simulation
    pars['n_days']     = 60           # Number of days to run, if end_day isn't specified
    pars['rand_seed']  = 1            # Random seed, if None, don't reset
    pars['verbose']    = 1            # Whether or not to display information during the run -- options are 0 (silent), 1 (default), 2 (everything)

    # Rescaling parameters
    pars['pop_scale']         = 1    # Factor by which to scale the population -- e.g. pop_scale=10 with pop_size=100e3 means a population of 1 million
    pars['rescale']           = True # Enable dynamic rescaling of the population -- starts with pop_scale=1 and scales up dynamically as the epidemic grows
    pars['rescale_threshold'] = 0.05 # Fraction susceptible population that will trigger rescaling if rescaling
    pars['rescale_factor']    = 1.2  # Factor by which the population is rescaled on each step

    # Basic disease transmission
    pars['beta']        = 0.016 # Beta per symptomatic contact; absolute value, calibrated
    pars['contacts']    = None  # The number of contacts per layer; set by reset_layer_pars() below
    pars['dynam_layer'] = None  # Which layers are dynamic; set by reset_layer_pars() below
    pars['beta_layer']  = None  # Transmissibility per layer; set by reset_layer_pars() below
    pars['n_imports']   = 0     # Average daily number of imported cases (actual number is drawn from Poisson distribution)
    pars['beta_dist']   = {'dist':'lognormal','par1':0.84, 'par2':0.3} # Distribution to draw individual level transmissibility; use 'neg_binomial' instead of 'lognormal' for more overdispersion; see https://wellcomeopenresearch.org/articles/5-67
    pars['viral_dist']  = {'frac_time':0.3, 'load_ratio':2, 'high_cap':4} # The time varying viral load (transmissibility); estimated from Lescure 2020, Lancet, https://doi.org/10.1016/S1473-3099(20)30200-0

    # Efficacy of protection measures
    pars['asymp_factor'] = 1.0 # Multiply beta by this factor for asymptomatic cases; no statistically significant difference in transmissibility: https://www.sciencedirect.com/science/article/pii/S1201971220302502
    pars['iso_factor']   = None # Multiply beta by this factor for diganosed cases to represent isolation; set by reset_layer_pars() below
    pars['quar_factor']  = None # Quarantine multiplier on transmissibility and susceptibility; set by reset_layer_pars() below
    pars['quar_period']  = 14  # Number of days to quarantine for; assumption based on standard policies

    # Duration parameters: time for disease progression
    pars['dur'] = {}
    pars['dur']['exp2inf']  = {'dist':'lognormal_int', 'par1':4.6, 'par2':4.8} # Duration from exposed to infectious; see Lauer et al., https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7081172/, subtracting inf2sim duration
    pars['dur']['inf2sym']  = {'dist':'lognormal_int', 'par1':1.0, 'par2':0.9} # Duration from infectious to symptomatic; see Linton et al., https://doi.org/10.3390/jcm9020538
    pars['dur']['sym2sev']  = {'dist':'lognormal_int', 'par1':6.6, 'par2':4.9} # Duration from symptomatic to severe symptoms; see Linton et al., https://doi.org/10.3390/jcm9020538
    pars['dur']['sev2crit'] = {'dist':'lognormal_int', 'par1':3.0, 'par2':7.4} # Duration from severe symptoms to requiring ICU; see Wang et al., https://jamanetwork.com/journals/jama/fullarticle/2761044

    # Duration parameters: time for disease recovery
    pars['dur']['asym2rec'] = {'dist':'lognormal_int', 'par1':8.0,  'par2':2.0} # Duration for asymptomatic people to recover; see Wölfel et al., https://www.nature.com/articles/s41586-020-2196-x
    pars['dur']['mild2rec'] = {'dist':'lognormal_int', 'par1':8.0,  'par2':2.0} # Duration for people with mild symptoms to recover; see Wölfel et al., https://www.nature.com/articles/s41586-020-2196-x
    pars['dur']['sev2rec']  = {'dist':'lognormal_int', 'par1':14.0, 'par2':2.4} # Duration for people with severe symptoms to recover, 22.6 days total; see Verity et al., https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf
    pars['dur']['crit2rec'] = {'dist':'lognormal_int', 'par1':14.0, 'par2':2.4} # Duration for people with critical symptoms to recover, 22.6 days total; see Verity et al., https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf
    pars['dur']['crit2die'] = {'dist':'lognormal_int', 'par1':6.2,  'par2':1.7} # Duration from critical symptoms to death, 17.8 days total; see Verity et al., https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf

    # Severity parameters: probabilities of symptom progression
    pars['rel_symp_prob']   = 1.0  # Scale factor for proportion of symptomatic cases
    pars['rel_severe_prob'] = 1.0  # Scale factor for proportion of symptomatic cases that become severe
    pars['rel_crit_prob']   = 1.0  # Scale factor for proportion of severe cases that become critical
    pars['rel_death_prob']  = 1.0  # Scale factor for proportion of critical cases that result in death
    pars['prog_by_age']     = prog_by_age # Whether to set disease progression based on the person's age
    pars['prognoses']       = None # The actual arrays of prognoses by age; this is populated later

    # Events and interventions
    pars['interventions'] = []   # The interventions present in this simulation; populated by the user
    pars['analyzers']     = []   # Custom analysis functions; populated by the user
    pars['timelimit']     = None # Time limit for the simulation (seconds)
    pars['stopping_func'] = None # A function to call to stop the sim partway through

    # Health system parameters
    pars['n_beds_hosp']    = None # The number of hospital (adult acute care) beds available for severely ill patients (default is no constraint)
    pars['n_beds_icu']     = None # The number of ICU beds available for critically ill patients (default is no constraint)
    pars['no_hosp_factor'] = 2.0  # Multiplier for how much more likely severely ill people are to become critical if no hospital beds are available
    pars['no_icu_factor']  = 2.0  # Multiplier for how much more likely critically ill people are to die if no ICU beds are available

    # Update with any supplied parameter values and generate things that need to be generated
    pars.update(kwargs)
    reset_layer_pars(pars)
    if set_prognoses: # If not set here, gets set when the population is initialized
        pars['prognoses'] = get_prognoses(pars['prog_by_age']) # Default to age-specific prognoses

    return pars


# Define which parametrs need to be specified as a dictionary by layer -- define here so it's available at the module level for sim.py
layer_pars = ['beta_layer', 'contacts', 'dynam_layer', 'iso_factor', 'quar_factor']


def reset_layer_pars(pars, layer_keys=None, force=False):
    '''
    Small helper function to set layer-specific parameters. If layer keys are not
    provided, then set them based on the population type.

    Args:
        pars (dict): the parameters dictionary
        layer_keys (list): the layer keys of the population, if available
        force (bool): reset the pars even if they already exist
    '''

    # Specify defaults for random -- layer 'a' for 'all'
    defaults_r = dict(
        beta_layer  = dict(a=1.0), # Default beta
        contacts    = dict(a=20),  # Default number of contacts
        dynam_layer = dict(a=0),   # Do not use dynamic layers by default
        iso_factor  = dict(a=0.2), # Assumed isolation factor
        quar_factor = dict(a=0.3), # Assumed quarantine factor
    )

    # Specify defaults for hybrid (and SynthPops) -- household, school, work, and community layers (h, s, w, c)
    defaults_h = dict(
        beta_layer  = dict(h=7.0, s=0.7, w=0.7, c=0.14), # Per-population beta weights; relative
        contacts    = dict(h=2.0, s=20,  w=16,  c=20),   # Number of contacts per person per day, estimated
        dynam_layer = dict(h=0,   s=0,   w=0,   c=0),    # Which layers are dynamic -- none by default
        iso_factor  = dict(h=0.3, s=0.0, w=0.0, c=0.1),  # Multiply beta by this factor for people in isolation
        quar_factor = dict(h=0.8, s=0.0, w=0.0, c=0.3),  # Multiply beta by this factor for people in quarantine
    )

    # Choose the parameter defaults based on the population type, and get the layer keys
    if pars['pop_type'] == 'random':
        defaults = defaults_r
        default_layer_keys = ['a'] # Although this could be retrieved from the dictionary, make it explicit
    else:
        defaults = defaults_h
        default_layer_keys = ['h', 's', 'w', 'c'] # NB, these must match defaults_h above

    # Actually set the parameters
    for pkey in layer_pars:
        par = {} # Initialize this parameter
        default_val = defaults_r[pkey]['a'] # Get the default value for this parameter

        # If forcing, we overwrite any existing parameter values
        if force:
            par_dict = defaults[pkey] # Just use defaults
        else:
            par_dict = sc.mergedicts(defaults[pkey], pars.get(pkey, None)) # Use user-supplied parameters if available, else default

        # Figure out what the layer keys for this parameter are (may be different between parameters)
        if layer_keys:
            par_layer_keys = layer_keys # Use supplied layer keys
        else:
            par_layer_keys = list(sc.odict.fromkeys(default_layer_keys + list(par_dict.keys())))  # If not supplied, use the defaults, plus any extra from the par_dict; adapted from https://www.askpython.com/python/remove-duplicate-elements-from-list-python

        # Construct this parameter, layer by layer
        for lkey in par_layer_keys: # Loop over layers
            par[lkey] = par_dict.get(lkey, default_val) # Get the value for this layer if available, else use the default for random
        pars[pkey] = par # Save this parameter to the dictionary

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

    if not by_age:
        prognoses = dict(
            age_cutoffs  = np.array([0]),
            symp_probs   = np.array([0.75]),
            severe_probs = np.array([0.20]),
            crit_probs   = np.array([0.08]),
            death_probs  = np.array([0.02]),
        )
    else:
        prognoses = dict(
            age_cutoffs   = np.array([0,       10,      20,      30,      40,      50,      60,      70,      80,      90,]),     # Age cutoffs (lower limits)
            sus_ORs       = np.array([0.34,    0.67,    1.00,    1.00,    1.00,    1.00,    1.24,    1.47,    1.47,    1.47]),    # Odds ratios for relative susceptibility -- from https://science.sciencemag.org/content/early/2020/05/04/science.abb8001; 10-20 and 60-70 bins are the average across the ORs
            trans_ORs     = np.array([1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00]),    # Odds ratios for relative transmissibility -- no evidence of differences
            comorbidities = np.array([1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00,    1.00]),    # Comorbidities by age -- set to 1 by default since already included in disease progression rates
            symp_probs    = np.array([0.50,    0.55,    0.60,    0.65,    0.70,    0.75,    0.80,    0.85,    0.90,    0.90]),    # Overall probability of developing symptoms (based on https://www.medrxiv.org/content/10.1101/2020.03.24.20043018v1.full.pdf, scaled for overall symptomaticity)
            severe_probs  = np.array([0.00050, 0.00165, 0.00720, 0.02080, 0.03430, 0.07650, 0.13280, 0.20655, 0.24570, 0.24570]), # Overall probability of developing severe symptoms (derived from Table 1 of https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf)
            crit_probs    = np.array([0.00003, 0.00008, 0.00036, 0.00104, 0.00216, 0.00933, 0.03639, 0.08923, 0.17420, 0.17420]), # Overall probability of developing critical symptoms (derived from Table 1 of https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf)
            death_probs   = np.array([0.00002, 0.00006, 0.00030, 0.00080, 0.00150, 0.00600, 0.02200, 0.05100, 0.09300, 0.09300]), # Overall probability of dying (https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf)
        )

    prognoses['death_probs']  /= prognoses['crit_probs']   # Conditional probability of dying, given critical symptoms
    prognoses['crit_probs']   /= prognoses['severe_probs'] # Conditional probability of symptoms becoming critical, given severe
    prognoses['severe_probs'] /= prognoses['symp_probs']   # Conditional probability of symptoms becoming severe, given symptomatic

    return prognoses
