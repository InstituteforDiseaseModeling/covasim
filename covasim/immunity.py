'''
Defines classes and methods for calculating immunity
'''

import numpy as np
import sciris as sc
from . import utils as cvu
from . import defaults as cvd
from . import parameters as cvpar
from . import interventions as cvi


# %% Define variant class -- all other functions are for internal use only

__all__ = ['variant']


class variant(sc.prettyobj):
    '''
    Add a new variant to the sim

    Args:
        variant (str/dict): name of variant, or dictionary of parameters specifying information about the variant
        days   (int/list): day(s) on which new variant is introduced
        label       (str): if variant is supplied as a dict, the name of the variant
        n_imports   (int): the number of imports of the variant to be added
        rescale    (bool): whether the number of imports should be rescaled with the population

    **Example**::

        b117    = cv.variant('b117', days=10) # Make variant B117 active from day 10
        p1      = cv.variant('p1', days=15) # Make variant P1 active from day 15
        my_var  = cv.variant(variant={'rel_beta': 2.5}, label='My variant', days=20)
        sim     = cv.Sim(variants=[b117, p1, my_var]).run() # Add them all to the sim
        sim2    = cv.Sim(variants=cv.variant('b117', days=0, n_imports=20), pop_infected=0).run() # Replace default variant with b117
    '''

    def __init__(self, variant, days, label=None, n_imports=1, rescale=True):
        self.days = days # Handle inputs
        self.n_imports = int(n_imports)
        self.rescale   = rescale
        self.index     = None # Index of the variant in the sim; set later
        self.label     = None # Variant label (used as a dict key)
        self.p         = None # This is where the parameters will be stored
        self.parse(variant=variant, label=label) # Variants can be defined in different ways: process these here
        self.initialized = False
        return


    def parse(self, variant=None, label=None):
        ''' Unpack variant information, which may be given as either a string or a dict '''

        # Option 1: variants can be chosen from a list of pre-defined variants
        if isinstance(variant, str):

            choices, mapping = cvpar.get_variant_choices()
            known_variant_pars = cvpar.get_variant_pars()

            label = variant.lower()
            for txt in ['.', ' ', 'variant', 'variant', 'voc']:
                label = label.replace(txt, '')

            if label in mapping:
                label = mapping[label]
                variant_pars = known_variant_pars[label]
            else:
                errormsg = f'The selected variant "{variant}" is not implemented; choices are:\n{sc.pp(choices, doprint=False)}'
                raise NotImplementedError(errormsg)

        # Option 2: variants can be specified as a dict of pars
        elif isinstance(variant, dict):

            default_variant_pars = cvpar.get_variant_pars(default=True)
            default_keys = list(default_variant_pars.keys())

            # Parse label
            variant_pars = variant
            label = variant_pars.pop('label', label) # Allow including the label in the parameters
            if label is None:
                label = 'custom'

            # Check that valid keys have been supplied...
            invalid = []
            for key in variant_pars.keys():
                if key not in default_keys:
                    invalid.append(key)
            if len(invalid):
                errormsg = f'Could not parse variant keys "{sc.strjoin(invalid)}"; valid keys are: "{sc.strjoin(cvd.variant_pars)}"'
                raise sc.KeyNotFoundError(errormsg)

            # ...and populate any that are missing
            for key in default_keys:
                if key not in variant_pars:
                    variant_pars[key] = default_variant_pars[key]

        else:
            errormsg = f'Could not understand {type(variant)}, please specify as a dict or a predefined variant:\n{sc.pp(choices, doprint=False)}'
            raise ValueError(errormsg)

        # Set label and parameters
        self.label = label
        self.p = sc.objdict(variant_pars)

        return


    def initialize(self, sim):
        ''' Update variant info in sim '''
        self.days = cvi.process_days(sim, self.days) # Convert days into correct format
        sim['variant_pars'][self.label] = self.p  # Store the parameters
        self.index = list(sim['variant_pars'].keys()).index(self.label) # Find where we are in the list
        sim['variant_map'][self.index]  = self.label # Use that to populate the reverse mapping
        self.initialized = True
        return


    def apply(self, sim):
        ''' Introduce new infections with this variant '''
        for ind in cvi.find_day(self.days, sim.t, interv=self, sim=sim): # Time to introduce variant
            susceptible_inds = cvu.true(sim.people.susceptible)
            rescale_factor = sim.rescale_vec[sim.t] if self.rescale else 1.0
            n_imports = sc.randround(self.n_imports/rescale_factor) # Round stochastically to the nearest number of imports
            importation_inds = np.random.choice(susceptible_inds, n_imports)
            sim.people.infect(inds=importation_inds, layer='importation', variant=self.index)
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


def update_peak_nab(people, inds, nab_pars, natural=True):
    '''
    Update peak NAb level

    This function updates the peak NAb level for individuals when a NAb boosting event occurs.
    NAbs can come from a natural infection or vaccination and the peak level depends on if there
    is prior immunity:

    1) a natural infection. If individual has no existing NAb, draw from distribution
    depending upon symptoms. If individual has existing NAb, multiply booster impact
    2) Vaccination. If individual has no existing NAb, draw from distribution
    depending upon vaccine source. If individual has existing NAb, multiply booster impact

    Additionally, for people with no prior immunity and with natural infection, the peak NAb
    is scaled by whether or not the person develops symptoms.

    Args:
        people: A people object
        inds: Array of people indices
        nab_pars: Parameters from which to draw values for quantities like ['nab_init'] - either
                    sim pars (for natural immunity) or vaccine pars
        natural: If True, immunity is being changed due to infection rather than vaccination

    Returns: None
    '''


    has_nabs = people.nab[inds] > 0
    no_prior_nab_inds = inds[~has_nabs]
    prior_nab_inds = inds[has_nabs]

    # NAb from infection
    if natural:
        # 1) No prior NAb: draw NAb from a distribution and compute
        if len(no_prior_nab_inds):
            init_nab = cvu.sample(**nab_pars['nab_init'], size=len(no_prior_nab_inds))
            prior_symp = people.prior_symptoms[no_prior_nab_inds]
            no_prior_nab = (2**init_nab) * prior_symp
            people.peak_nab[no_prior_nab_inds] = no_prior_nab

        # 2) Prior NAb: multiply existing NAb by boost factor
        if len(prior_nab_inds):
            people.peak_nab[prior_nab_inds] *= nab_pars['nab_boost']

    # NAb from a vaccine
    else:
        # 1) No prior NAb: draw NAb from a distribution and compute
        if len(no_prior_nab_inds):
            init_nab = cvu.sample(**nab_pars['nab_init'], size=len(no_prior_nab_inds))
            people.peak_nab[no_prior_nab_inds] = 2**init_nab

        # 2) Prior nab (from natural or vaccine dose 1): multiply existing nab by boost factor
        if len(prior_nab_inds):
            people.peak_nab[prior_nab_inds] *= nab_pars['nab_boost']

    return


def update_nab(people, inds):
    '''
    Step NAb levels forward in time
    '''
    t_since_boost = people.t-np.fmax(people.date_exposed[inds],people.date_vaccinated[inds]).astype(cvd.default_int)
    people.nab[inds] += people.pars['nab_kin'][t_since_boost]*people.peak_nab[inds]
    people.nab[inds[people.nab[inds] < 0]] = 0 # Make sure nabs don't drop below 0
    people.nab[inds[people.nab[inds] > people.peak_nab[inds]]] = people.peak_nab[inds[people.nab[inds] > people.peak_nab[inds]]] # Make sure nabs don't exceed peak_nab
    return


def calc_VE(alpha_inf, beta_inf, nab, **kwargs):
    ''' Calculate vaccine efficacy based on the vaccine parameters and NAbs '''
    zero_nab    = nab == 0 # To avoid taking logarithm of 0
    nonzero_nab = nab > 0
    lo = alpha_inf + beta_inf*np.log(nab, where=nonzero_nab)
    exp_lo = np.exp(lo, where=nonzero_nab)
    exp_lo[zero_nab] = 0 # Re-insert zeros
    output = exp_lo/(1+exp_lo) # Inverse logit function
    return output


def calc_VE_symp(alpha_inf, beta_inf, alpha_symp_inf, beta_symp_inf, nab, **kwargs):
    ''' As above, for symptoms given infection '''
    inf_VE  = calc_VE(alpha_inf,      beta_inf,      nab)
    symp_VE = calc_VE(alpha_symp_inf, beta_symp_inf, nab)
    output  = 1 - (1-inf_VE) * (1-symp_VE)
    return output


def calc_VE_sev(alpha_inf, beta_inf, alpha_symp_inf, beta_symp_inf, alpha_sev_symp, beta_sev_symp, nab, **kwargs):
    ''' As above, for severe disease '''
    inf_VE  = calc_VE(alpha_inf,      beta_inf,      nab)
    symp_VE = calc_VE(alpha_symp_inf, beta_symp_inf, nab)
    sev_VE  = calc_VE(alpha_sev_symp, beta_sev_symp, nab)
    output  = 1 - (1-inf_VE) * (1-symp_VE) * (1-sev_VE)
    return output


def calc_VE_symp_inf(alpha_inf, beta_inf, alpha_symp_inf, beta_symp_inf, nab, **kwargs):
    ''' As above, for symptoms and infection '''
    VE_inf  = calc_VE(alpha_inf, beta_inf, nab)
    VE_symp = calc_VE_symp(alpha_inf, beta_inf, alpha_symp_inf, beta_symp_inf, nab)
    output = 1 - ((1-VE_symp)/(1-VE_inf))
    return output


def calc_VE_sev_symp(alpha_inf, beta_inf, alpha_symp_inf, beta_symp_inf, alpha_sev_symp, beta_sev_symp, nab, **kwargs):
    ''' As above, for severe disease '''
    VE_inf      = calc_VE(alpha_inf, beta_inf, nab)
    VE_symp_inf = calc_VE_symp_inf(alpha_inf, beta_inf, alpha_symp_inf, beta_symp_inf, nab)
    VE_sev      = calc_VE_sev(alpha_inf, beta_inf, alpha_symp_inf, beta_symp_inf, alpha_sev_symp, beta_sev_symp, nab)
    output      = 1 - ((1-VE_sev)/((1-VE_inf)*(1-VE_symp_inf)))
    return output


def nab_to_efficacy(nab, ax, pars):
    '''
    Convert NAb levels to immunity protection factors, using the functional form
    given in this paper: https://doi.org/10.1101/2021.03.09.21252641

    Args:
        nab (arr): an array of effective NAb levels (i.e. actual NAb levels, scaled by cross-immunity)
        ax (str): can be 'sus', 'symp' or 'sev', corresponding to the efficacy of protection against infection, symptoms, and severe disease respectively
        pars (dict): dictionary of parameters for the vaccine efficacy

    Returns:
        an array the same size as NAb, containing the immunity protection factors for the specified axis
     '''
    choices = ['sus', 'symp', 'sev']
    if ax not in choices:
        errormsg = f'Choice {ax} not in list of choices: {sc.strjoin(choices)}'
        raise ValueError(errormsg)
    if   ax == 'sus':  efficacy = calc_VE(nab=nab, **pars)
    elif ax == 'symp': efficacy = calc_VE_symp_inf(nab=nab, **pars)
    elif ax == 'sev':  efficacy = calc_VE_sev_symp(nab=nab, **pars)
    return efficacy



# %% Immunity methods

def init_immunity(sim, create=False):
    ''' Initialize immunity matrices with all variants that will eventually be in the sim'''

    # Don't use this function if immunity is turned off
    if not sim['use_waning']:
        return

    # Pull out all of the circulating variants for cross-immunity
    ns = sim['n_variants']

    # If immunity values have been provided, process them
    if sim['immunity'] is None or create:

        # Firstly, initialize immunity matrix with defaults. These are then overwitten with variant-specific values below
        # Susceptibility matrix is of size sim['n_variants']*sim['n_variants']
        immunity = np.ones((ns, ns), dtype=cvd.default_float)  # Fill with defaults

        # Next, overwrite these defaults with any known immunity values about specific variants
        default_cross_immunity = cvpar.get_cross_immunity()
        for i in range(ns):
            label_i = sim['variant_map'][i]
            for j in range(ns):
                label_j = sim['variant_map'][j]
                if label_i in default_cross_immunity and label_j in default_cross_immunity:
                    immunity[j][i] = default_cross_immunity[label_j][label_i]

        sim['immunity'] = immunity

    # Next, precompute the NAb kinetics and store these for access during the sim
    sim['nab_kin'] = precompute_waning(length=sim.npts, pars=sim['nab_decay'])

    return


def check_immunity(people, variant, sus=True, inds=None):
    '''
    Calculate people's immunity on this timestep from prior infections + vaccination

    There are two fundamental sources of immunity:

           (1) prior exposure: degree of protection depends on variant, prior symptoms, and time since recovery
           (2) vaccination: degree of protection depends on variant, vaccine, and time since vaccination

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
        vacc_mapping = np.array([vaccine_pars.get(label, 1.0) for label in pars['variant_map'].values()]) # TODO: make more robust

    # PART 1: Immunity to infection for susceptible individuals
    if sus:
        is_sus = cvu.true(people.susceptible)  # Currently susceptible
        was_inf_same = cvu.true((people.recovered_variant == variant) & (people.t >= date_rec))  # Had a previous exposure to the same variant, now recovered
        was_inf_diff = np.setdiff1d(was_inf, was_inf_same)  # Had a previous exposure to a different variant, now recovered
        is_sus_vacc = np.intersect1d(is_sus, is_vacc)  # Susceptible and vaccinated
        is_sus_vacc = np.setdiff1d(is_sus_vacc, was_inf)  # Susceptible, vaccinated without prior infection
        is_sus_was_inf_same = np.intersect1d(is_sus, was_inf_same)  # Susceptible and being challenged by the same variant
        is_sus_was_inf_diff = np.intersect1d(is_sus, was_inf_diff)  # Susceptible and being challenged by a different variant

        if len(is_sus_vacc):
            vaccine_source = cvd.default_int(people.vaccine_source[is_sus_vacc]) # TODO: use vaccine source
            vaccine_scale = vacc_mapping[variant]
            current_nabs = people.nab[is_sus_vacc]
            people.sus_imm[variant, is_sus_vacc] = nab_to_efficacy(current_nabs * vaccine_scale, 'sus', vx_nab_eff_pars)

        if len(is_sus_was_inf_same):  # Immunity for susceptibles with prior exposure to this variant
            current_nabs = people.nab[is_sus_was_inf_same]
            people.sus_imm[variant, is_sus_was_inf_same] = nab_to_efficacy(current_nabs * immunity[variant, variant], 'sus', nab_eff)

        if len(is_sus_was_inf_diff):  # Cross-immunity for susceptibles with prior exposure to a different variant
            prior_variants = people.recovered_variant[is_sus_was_inf_diff]
            prior_variants_unique = cvd.default_int(np.unique(prior_variants))
            for unique_variant in prior_variants_unique:
                unique_inds = is_sus_was_inf_diff[cvu.true(prior_variants == unique_variant)]
                current_nabs = people.nab[unique_inds]
                people.sus_imm[variant, unique_inds] = nab_to_efficacy(current_nabs * immunity[variant, unique_variant], 'sus', nab_eff)

    # PART 2: Immunity to disease for currently-infected people
    else:
        is_inf_vacc = np.intersect1d(inds, is_vacc)
        was_inf = np.intersect1d(inds, was_inf)

        if len(is_inf_vacc):  # Immunity for infected people who've been vaccinated
            vaccine_source = cvd.default_int(people.vaccine_source[is_inf_vacc])  # TODO: use vaccine source
            vaccine_scale = vacc_mapping[variant]
            current_nabs = people.nab[is_inf_vacc]
            people.symp_imm[variant, is_inf_vacc] = nab_to_efficacy(current_nabs * vaccine_scale, 'symp', nab_eff)
            people.sev_imm[variant, is_inf_vacc] = nab_to_efficacy(current_nabs * vaccine_scale, 'sev', nab_eff)

        if len(was_inf):  # Immunity for reinfected people
            current_nabs = people.nab[was_inf]
            people.symp_imm[variant, was_inf] = nab_to_efficacy(current_nabs, 'symp', nab_eff)
            people.sev_imm[variant, was_inf] = nab_to_efficacy(current_nabs, 'sev', nab_eff)

    return



#%% Methods for computing waning

def precompute_waning(length, pars=None):
    '''
    Process functional form and parameters into values:

        - 'nab_growth_decay' : based on Khoury et al. (https://www.nature.com/articles/s41591-021-01377-8)
        - 'nab_decay'   : specific decay function taken from https://doi.org/10.1101/2021.03.09.21252641
        - 'exp_decay'   : exponential decay. Parameters should be init_val and half_life (half_life can be None/nan)
        - 'linear_decay': linear decay

    A custom function can also be supplied.

    Args:
        length (float): length of array to return, i.e., for how long waning is calculated
        pars (dict): passed to individual immunity functions

    Returns:
        array of length 'length' of values
    '''

    pars = sc.dcp(pars)
    form = pars.pop('form')
    choices = [
        'nab_growth_decay', # Default if no form is provided
        'nab_decay',
        'exp_decay',
    ]

    # Process inputs
    if form is None or form == 'nab_growth_decay':
        output = nab_growth_decay(length, **pars)

    elif form == 'nab_decay':
        output = nab_decay(length, **pars)

    elif form == 'exp_decay':
        if pars['half_life'] is None: pars['half_life'] = np.nan
        output = exp_decay(length, **pars)

    elif callable(form):
        output = form(length, **pars)

    else:
        errormsg = f'The selected functional form "{form}" is not implemented; choices are: {sc.strjoin(choices)}'
        raise NotImplementedError(errormsg)

    return output


def nab_growth_decay(length, growth_time, decay_rate1, decay_time1, decay_rate2, decay_time2):
    '''
    Returns an array of length 'length' containing the evaluated function nab growth/decay
    function at each point.

    Uses linear growth + exponential decay, with the rate of exponential decay also set to
    decay linearly until it reaches a 10-year half life.

    Args:
        length (int): number of points
        growth_time (int): length of time NAbs grow (used to determine slope)
        decay_rate1 (float): initial rate of exponential decay
        decay_time1 (float): time on the first exponential decay
        decay_rate2 (float): the rate of exponential decay in late period
        decay_time2 (float): how long it takes to transition to late decay period
    '''


    def f1(t, growth_time):
        '''Simple linear growth'''
        return (1 / growth_time) * t

    def f2(t, decay_time1, decay_time2, decay_rate1, decay_rate2):
        decayRate = np.full(len(t), fill_value=decay_rate1)
        decayRate[cvu.true(t>decay_time2)] = decay_rate2
        slowing = (1 / (decay_time2 - decay_time1)) * (decay_rate1 - decay_rate2)
        decayRate[cvu.true((t>decay_time1)*(t<=decay_time2))] = decay_rate1 - slowing * (np.arange(len(cvu.true((t>decay_time1)*(t<=decay_time2))), dtype=cvd.default_int))
        titre = np.zeros(len(t))
        for i in range(1, len(t)):
            titre[i] = titre[i-1]+decayRate[i]
        return np.exp(-titre)

    length = length + 1
    t1 = np.arange(growth_time, dtype=cvd.default_int)
    t2 = np.arange(length - growth_time, dtype=cvd.default_int)
    y1 = f1(t1, growth_time)
    y2 = f2(t2, decay_time1, decay_time2, decay_rate1, decay_rate2)
    y  = np.concatenate([y1,y2])
    y = np.diff(y)[0:length]

    return y


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
    y  = np.concatenate([[-np.inf],y1,y2])
    y = np.diff(y)[0:length]
    y[0] = 1
    return y


def exp_decay(length, init_val, half_life, delay=None):
    '''
    Returns an array of length t with values for the immunity at each time step after recovery
    '''
    length = length+1
    decay_rate = np.log(2) / half_life if ~np.isnan(half_life) else 0.
    if delay is not None:
        t = np.arange(length-delay, dtype=cvd.default_int)
        growth = linear_growth(delay, init_val/delay)
        decay = init_val * np.exp(-decay_rate * t)
        result = np.concatenate([growth, decay], axis=None)
    else:
        t = np.arange(length, dtype=cvd.default_int)
        result = init_val * np.exp(-decay_rate * t)
    return np.diff(result)


def linear_decay(length, init_val, slope):
    ''' Calculate linear decay '''
    result = -slope*np.ones(length)
    result[0] = init_val
    return result


def linear_growth(length, slope):
    ''' Calculate linear growth '''
    return slope*np.ones(length)
