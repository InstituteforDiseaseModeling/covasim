'''
Defines classes and methods for calculating immunity
'''

import numpy as np
import sciris as sc
from . import utils as cvu
from . import defaults as cvd
from . import parameters as cvpar
from . import interventions as cvi


# %% Define strain class -- all other functions are for internal use only

__all__ = ['strain']


class strain(sc.prettyobj):
    '''
    Add a new strain to the sim

    Args:
        strain (str/dict): name of strain, or dictionary of parameters specifying information about the strain
        days   (int/list): day(s) on which new variant is introduced
        label       (str): if strain is supplied as a dict, the name of the strain
        n_imports   (int): the number of imports of the strain to be added
        rescale    (bool): whether the number of imports should be rescaled with the population

    **Example**::

        b117    = cv.strain('b117', days=10) # Make strain B117 active from day 10
        p1      = cv.strain('p1', days=15) # Make strain P1 active from day 15
        my_var  = cv.strain(strain={'rel_beta': 2.5}, label='My strain', days=20)
        sim     = cv.Sim(strains=[b117, p1, my_var]).run() # Add them all to the sim
        sim2    = cv.Sim(strains=cv.strain('b117', days=0, n_imports=20), pop_infected=0).run() # Replace default strain with b117
    '''

    def __init__(self, strain, days, label=None, n_imports=1, rescale=True):
        self.days = days # Handle inputs
        self.n_imports = int(n_imports)
        self.rescale   = rescale
        self.index     = None # Index of the strain in the sim; set later
        self.label     = None # Strain label (used as a dict key)
        self.p         = None # This is where the parameters will be stored
        self.parse(strain=strain, label=label) # Strains can be defined in different ways: process these here
        self.initialized = False
        return


    def parse(self, strain=None, label=None):
        ''' Unpack strain information, which may be given as either a string or a dict '''

        # Option 1: strains can be chosen from a list of pre-defined strains
        if isinstance(strain, str):

            choices, mapping = cvpar.get_strain_choices()
            known_strain_pars = cvpar.get_strain_pars()

            label = strain.lower()
            for txt in ['.', ' ', 'strain', 'variant', 'voc']:
                label = label.replace(txt, '')

            if label in mapping:
                label = mapping[label]
                strain_pars = known_strain_pars[label]
            else:
                errormsg = f'The selected variant "{strain}" is not implemented; choices are:\n{sc.pp(choices, doprint=False)}'
                raise NotImplementedError(errormsg)

        # Option 2: strains can be specified as a dict of pars
        elif isinstance(strain, dict):

            default_strain_pars = cvpar.get_strain_pars(default=True)
            default_keys = list(default_strain_pars.keys())

            # Parse label
            strain_pars = strain
            label = strain_pars.pop('label', label) # Allow including the label in the parameters
            if label is None:
                label = 'custom'

            # Check that valid keys have been supplied...
            invalid = []
            for key in strain_pars.keys():
                if key not in default_keys:
                    invalid.append(key)
            if len(invalid):
                errormsg = f'Could not parse strain keys "{sc.strjoin(invalid)}"; valid keys are: "{sc.strjoin(cvd.strain_pars)}"'
                raise sc.KeyNotFoundError(errormsg)

            # ...and populate any that are missing
            for key in default_keys:
                if key not in strain_pars:
                    strain_pars[key] = default_strain_pars[key]

        else:
            errormsg = f'Could not understand {type(strain)}, please specify as a dict or a predefined strain:\n{sc.pp(choices, doprint=False)}'
            raise ValueError(errormsg)

        # Set label and parameters
        self.label = label
        self.p = sc.objdict(strain_pars)

        return


    def initialize(self, sim):
        ''' Update strain info in sim '''
        self.days = cvi.process_days(sim, self.days) # Convert days into correct format
        sim['strain_pars'][self.label] = self.p  # Store the parameters
        self.index = list(sim['strain_pars'].keys()).index(self.label) # Find where we are in the list
        sim['strain_map'][self.index]  = self.label # Use that to populate the reverse mapping
        self.initialized = True
        return


    def apply(self, sim):
        ''' Introduce new infections with this strain '''
        for ind in cvi.find_day(self.days, sim.t, interv=self, sim=sim): # Time to introduce strain
            susceptible_inds = cvu.true(sim.people.susceptible)
            rescale_factor = sim.rescale_vec[sim.t] if self.rescale else 1.0
            n_imports = sc.randround(self.n_imports/rescale_factor) # Round stochastically to the nearest number of imports
            importation_inds = np.random.choice(susceptible_inds, n_imports)
            sim.people.infect(inds=importation_inds, layer='importation', strain=self.index)
        return




#%% Neutralizing antibody methods

def get_vaccine_pars(pars):
    '''
    Temporary helper function to get vaccine parameters; to be refactored

    TODO: use people.vaccine_source to get the per-person specific NAb decay
    '''
    try:
        vaccine = pars['vaccine_map'][0] # For now, just use the first vaccine, if available
        vaccine_pars = pars['vaccine_pars'][vaccine]
    except:
        vaccine_pars = pars # Otherwise, just use defaults for natural immunity

    return vaccine_pars


def init_nab(people, inds, prior_inf=True):
    '''
    Draws an initial neutralizing antibody (NAb) level for individuals.
    Can come from a natural infection or vaccination and depends on if there is prior immunity:
    1) a natural infection. If individual has no existing NAb, draw from distribution
    depending upon symptoms. If individual has existing NAb, multiply booster impact
    2) Vaccination. If individual has no existing NAb, draw from distribution
    depending upon vaccine source. If individual has existing NAb, multiply booster impact
    '''

    nab_arrays = people.nab[inds]
    prior_nab_inds = cvu.idefined(nab_arrays, inds) # Find people with prior NAb
    no_prior_nab_inds = np.setdiff1d(inds, prior_nab_inds) # Find people without prior NAb
    peak_nab = people.init_nab[prior_nab_inds]
    pars = people.pars

    # NAb from infection
    if prior_inf:
        nab_boost = pars['nab_boost']  # Boosting factor for natural infection
        # 1) No prior NAb: draw NAb from a distribution and compute
        if len(no_prior_nab_inds):
            init_nab = cvu.sample(**pars['nab_init'], size=len(no_prior_nab_inds))
            prior_symp = people.prior_symptoms[no_prior_nab_inds]
            no_prior_nab = (2**init_nab) * prior_symp
            people.init_nab[no_prior_nab_inds] = no_prior_nab

        # 2) Prior NAb: multiply existing NAb by boost factor
        if len(prior_nab_inds):
            init_nab = peak_nab * nab_boost
            people.init_nab[prior_nab_inds] = init_nab

    # NAb from a vaccine
    else:
        vaccine_pars = get_vaccine_pars(pars)

        # 1) No prior NAb: draw NAb from a distribution and compute
        if len(no_prior_nab_inds):
            init_nab = cvu.sample(**vaccine_pars['nab_init'], size=len(no_prior_nab_inds))
            people.init_nab[no_prior_nab_inds] = 2**init_nab

        # 2) Prior nab (from natural or vaccine dose 1): multiply existing nab by boost factor
        if len(prior_nab_inds):
            nab_boost = vaccine_pars['nab_boost']  # Boosting factor for vaccination
            init_nab = peak_nab * nab_boost
            people.init_nab[prior_nab_inds] = init_nab

    return


def check_nab(t, people, inds=None):
    ''' Determines current NAb based on date since recovered/vaccinated.'''

    # Indices of people who've had some nab event
    rec_inds = cvu.defined(people.date_recovered[inds])
    vac_inds = cvu.defined(people.date_vaccinated[inds])
    both_inds = np.intersect1d(rec_inds, vac_inds)

    # Time since boost
    t_since_boost = np.full(len(inds), np.nan, dtype=cvd.default_int)
    t_since_boost[rec_inds] = t-people.date_recovered[inds[rec_inds]]
    t_since_boost[vac_inds] = t-people.date_vaccinated[inds[vac_inds]]
    t_since_boost[both_inds] = t-np.maximum(people.date_recovered[inds[both_inds]],people.date_vaccinated[inds[both_inds]])

    # Set current NAb
    people.nab[inds] = people.pars['nab_kin'][t_since_boost] * people.init_nab[inds]

    return


def nab_to_efficacy(nab, ax, function_args):
    '''
    Convert NAb levels to immunity protection factors, using the functional form
    given in this paper: https://doi.org/10.1101/2021.03.09.21252641

    Args:
        nab (arr): an array of NAb levels
        ax (str): can be 'sus', 'symp' or 'sev', corresponding to the efficacy of protection against infection, symptoms, and severe disease respectively

    Returns:
        an array the same size as NAb, containing the immunity protection factors for the specified axis
     '''

    if ax not in ['sus', 'symp', 'sev']:
        errormsg = f'Choice {ax} not in list of choices'
        raise ValueError(errormsg)
    args = function_args[ax]

    if ax == 'sus':
        slope = args['slope']
        n_50 = args['n_50']
        efficacy = 1 / (1 + np.exp(-slope * (np.log10(nab) - np.log10(n_50))))  # from logistic regression computed in R using data from Khoury et al
    else:
        efficacy = np.full(len(nab), fill_value=args)
    return efficacy



# %% Immunity methods

def init_immunity(sim, create=False):
    ''' Initialize immunity matrices with all strains that will eventually be in the sim'''

    # Don't use this function if immunity is turned off
    if not sim['use_waning']:
        return

    # Pull out all of the circulating strains for cross-immunity
    ns       = sim['n_strains']

    # If immunity values have been provided, process them
    if sim['immunity'] is None or create:

        # Firstly, initialize immunity matrix with defaults. These are then overwitten with strain-specific values below
        # Susceptibility matrix is of size sim['n_strains']*sim['n_strains']
        immunity = np.ones((ns, ns), dtype=cvd.default_float)  # Fill with defaults

        # Next, overwrite these defaults with any known immunity values about specific strains
        default_cross_immunity = cvpar.get_cross_immunity()
        for i in range(ns):
            label_i = sim['strain_map'][i]
            for j in range(ns):
                if i != j: # Populate cross-immunity
                    label_j = sim['strain_map'][j]
                    if label_i in default_cross_immunity and label_j in default_cross_immunity:
                        immunity[j][i] = default_cross_immunity[label_j][label_i]
                else: # Populate own-immunity
                    immunity[i, i] = sim['strain_pars'][label_i]['rel_imm_strain']

        sim['immunity'] = immunity

    # Next, precompute the NAb kinetics and store these for access during the sim
    sim['nab_kin'] = precompute_waning(length=sim['n_days'], pars=sim['nab_decay'])

    return


def check_immunity(people, strain, sus=True, inds=None):
    '''
    Calculate people's immunity on this timestep from prior infections + vaccination

    There are two fundamental sources of immunity:

           (1) prior exposure: degree of protection depends on strain, prior symptoms, and time since recovery
           (2) vaccination: degree of protection depends on strain, vaccine, and time since vaccination

    Gets called from sim before computing trans_sus, sus=True, inds=None
    Gets called from people.infect() to calculate prog/trans, sus=False, inds= inds of people being infected
    '''

    # Handle parameters and indices
    pars = people.pars
    vaccine_pars = get_vaccine_pars(pars)
    was_inf  = cvu.true(people.t >= people.date_recovered)  # Had a previous exposure, now recovered
    is_vacc  = cvu.true(people.vaccinated)  # Vaccinated
    date_rec = people.date_recovered  # Date recovered
    immunity = pars['immunity'] # cross-immunity/own-immunity scalars to be applied to NAb level before computing efficacy
    nab_eff  = pars['nab_eff']

    # If vaccines are present, extract relevant information about them
    vacc_present = len(is_vacc)
    if vacc_present:
        vx_nab_eff_pars = vaccine_pars['nab_eff']
        vacc_mapping = np.array([vaccine_pars.get(label, 1.0) for label in pars['strain_map'].values()]) # TODO: make more robust

    # PART 1: Immunity to infection for susceptible individuals
    if sus:
        is_sus = cvu.true(people.susceptible)  # Currently susceptible
        was_inf_same = cvu.true((people.recovered_strain == strain) & (people.t >= date_rec))  # Had a previous exposure to the same strain, now recovered
        was_inf_diff = np.setdiff1d(was_inf, was_inf_same)  # Had a previous exposure to a different strain, now recovered
        is_sus_vacc = np.intersect1d(is_sus, is_vacc)  # Susceptible and vaccinated
        is_sus_vacc = np.setdiff1d(is_sus_vacc, was_inf)  # Susceptible, vaccinated without prior infection
        is_sus_was_inf_same = np.intersect1d(is_sus, was_inf_same)  # Susceptible and being challenged by the same strain
        is_sus_was_inf_diff = np.intersect1d(is_sus, was_inf_diff)  # Susceptible and being challenged by a different strain

        if len(is_sus_vacc):
            vaccine_source = cvd.default_int(people.vaccine_source[is_sus_vacc]) # TODO: use vaccine source
            vaccine_scale = vacc_mapping[strain]
            current_nabs = people.nab[is_sus_vacc]
            people.sus_imm[strain, is_sus_vacc] = nab_to_efficacy(current_nabs * vaccine_scale, 'sus', vx_nab_eff_pars)

        if len(is_sus_was_inf_same):  # Immunity for susceptibles with prior exposure to this strain
            current_nabs = people.nab[is_sus_was_inf_same]
            people.sus_imm[strain, is_sus_was_inf_same] = nab_to_efficacy(current_nabs * immunity[strain, strain], 'sus', nab_eff)

        if len(is_sus_was_inf_diff):  # Cross-immunity for susceptibles with prior exposure to a different strain
            prior_strains = people.recovered_strain[is_sus_was_inf_diff]
            prior_strains_unique = cvd.default_int(np.unique(prior_strains))
            for unique_strain in prior_strains_unique:
                unique_inds = is_sus_was_inf_diff[cvu.true(prior_strains == unique_strain)]
                current_nabs = people.nab[unique_inds]
                people.sus_imm[strain, unique_inds] = nab_to_efficacy(current_nabs * immunity[strain, unique_strain], 'sus', nab_eff)

    # PART 2: Immunity to disease for currently-infected people
    else:
        is_inf_vacc = np.intersect1d(inds, is_vacc)
        was_inf = np.intersect1d(inds, was_inf)

        if len(is_inf_vacc):  # Immunity for infected people who've been vaccinated
            vaccine_source = cvd.default_int(people.vaccine_source[is_inf_vacc])  # TODO: use vaccine source
            vaccine_scale = vacc_mapping[strain]
            current_nabs = people.nab[is_inf_vacc]
            people.symp_imm[strain, is_inf_vacc] = nab_to_efficacy(current_nabs * vaccine_scale, 'symp', nab_eff)
            people.sev_imm[strain, is_inf_vacc] = nab_to_efficacy(current_nabs * vaccine_scale, 'sev', nab_eff)

        if len(was_inf):  # Immunity for reinfected people
            current_nabs = people.nab[was_inf]
            people.symp_imm[strain, was_inf] = nab_to_efficacy(current_nabs, 'symp', nab_eff)
            people.sev_imm[strain, was_inf] = nab_to_efficacy(current_nabs, 'sev', nab_eff)

    return



#%% Methods for computing waning

def precompute_waning(length, pars=None):
    '''
    Process functional form and parameters into values:

        - 'nab_decay'   : specific decay function taken from https://doi.org/10.1101/2021.03.09.21252641
        - 'exp_decay'   : exponential decay. Parameters should be init_val and half_life (half_life can be None/nan)
        - 'linear_decay': linear decay

    Args:
        length (float): length of array to return, i.e., for how long waning is calculated
        pars (dict): passed to individual immunity functions

    Returns:
        array of length 'length' of values
    '''

    pars = sc.dcp(pars)
    form = pars.pop('form')
    choices = [
        'nab_decay', # Default if no form is provided
        'exp_decay',
        'linear_growth',
        'linear_decay'
    ]

    # Process inputs
    if form is None or form == 'nab_decay':
        output = nab_decay(length, **pars)

    elif form == 'exp_decay':
        if pars['half_life'] is None: pars['half_life'] = np.nan
        output = exp_decay(length, **pars)

    elif form == 'linear_growth':
        output = linear_growth(length, **pars)

    elif form == 'linear_decay':
        output = linear_decay(length, **pars)

    else:
        errormsg = f'The selected functional form "{form}" is not implemented; choices are: {sc.strjoin(choices)}'
        raise NotImplementedError(errormsg)

    return output


def nab_decay(length, decay_rate1, decay_time1, decay_rate2):
    '''
    Returns an array of length 'length' containing the evaluated function nab decay
    function at each point.

    Uses exponential decay, with the rate of exponential decay also set to exponentially
    decay (!) after 250 days.

    Args:
        length (int): number of points
        decay_rate1 (float): initial rate of exponential decay
        decay_time1 (float): time on the first exponential decay
        decay_rate2 (float): the rate at which the decay decays
    '''
    def f1(t, decay_rate1):
        ''' Simple exponential decay '''
        return np.exp(-t*decay_rate1)

    def f2(t, decay_rate1, decay_time1, decay_rate2):
        ''' Complex exponential decay '''
        return np.exp(-t*(decay_rate1*np.exp(-(t-decay_time1)*decay_rate2)))

    t  = np.arange(length, dtype=cvd.default_int)
    y1 = f1(cvu.true(t<=decay_time1), decay_rate1)
    y2 = f2(cvu.true(t>decay_time1), decay_rate1, decay_time1, decay_rate2)
    y  = np.concatenate([y1,y2])

    return y


def exp_decay(length, init_val, half_life, delay=None):
    '''
    Returns an array of length t with values for the immunity at each time step after recovery
    '''
    decay_rate = np.log(2) / half_life if ~np.isnan(half_life) else 0.
    if delay is not None:
        t = np.arange(length-delay, dtype=cvd.default_int)
        growth = linear_growth(delay, init_val/delay)
        decay = init_val * np.exp(-decay_rate * t)
        result = np.concatenate([growth, decay], axis=None)
    else:
        t = np.arange(length, dtype=cvd.default_int)
        result = init_val * np.exp(-decay_rate * t)
    return result


def linear_decay(length, init_val, slope):
    ''' Calculate linear decay '''
    t = np.arange(length, dtype=cvd.default_int)
    result = init_val - slope*t
    result = np.maximum(result, 0)
    return result


def linear_growth(length, slope):
    ''' Calculate linear growth '''
    t = np.arange(length, dtype=cvd.default_int)
    return (slope * t)
