'''
Defines classes and methods for calculating immunity
'''

import numpy as np
import pylab as pl
import sciris as sc
import datetime as dt
from . import utils as cvu
from . import defaults as cvd
from . import base as cvb
from . import parameters as cvpar
from . import people as cvppl
from collections import defaultdict

__all__ = []

# %% Define strain class

__all__ += ['Strain', 'Vaccine']


class Strain():
    '''
    Add a new strain to the sim

    Args:
        day (int): day on which new variant is introduced.
        n_imports (int): the number of imports of the strain to be added
        strain (dict): dictionary of parameters specifying information about the strain
        kwargs (dict):

    **Example**::
        b117    = cv.Strain('b117', days=10) # Make strain B117 active from day 10
        p1      = cv.Strain('p1', days=15) # Make strain P1 active from day 15
        my_var  = cv.Strain(strain={'rel_beta': 2.5}, strain_label='My strain', days=20) # Make a custom strain active from day 20
        sim     = cv.Sim(strains=[b117, p1, my_var]) # Add them all to the sim
    '''

    def __init__(self, strain=None, strain_label=None, days=None, n_imports=1, **kwargs):

        # Handle inputs
        self.days = days
        self.n_imports = n_imports

        # Strains can be defined in different ways: process these here
        self.strain_pars = self.parse_strain_pars(strain=strain, strain_label=strain_label)
        for par, val in self.strain_pars.items():
            setattr(self, par, val)
        return

    def parse_strain_pars(self, strain=None, strain_label=None):
        ''' Unpack strain information, which may be given in different ways'''

        # Option 1: strains can be chosen from a list of pre-defined strains
        if isinstance(strain, str):

            # List of choices currently available: new ones can be added to the list along with their aliases
            choices = {
                'wild': ['default', 'wild', 'pre-existing'],
                'b117': ['b117', 'B117', 'B.1.1.7', 'UK', 'uk', 'UK variant', 'uk variant'],
                'b1351': ['b1351', 'B1351', 'B.1.351', 'SA', 'sa', 'SA variant', 'sa variant'],
                # TODO: add other aliases
                'p1': ['p1', 'P1', 'P.1', 'B.1.1.248', 'b11248', 'Brazil', 'Brazil variant', 'brazil variant'],
            }

            # Empty pardict for wild strain
            if strain in choices['wild']:
                strain_pars = dict()
                self.strain_label = strain

            # Known parameters on B117
            elif strain in choices['b117']:
                strain_pars = dict()
                strain_pars['rel_beta'] = 1.5  # Transmissibility estimates range from 40-80%, see https://cmmid.github.io/topics/covid19/uk-novel-variant.html, https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2020.26.1.2002106
                strain_pars['rel_severe_prob'] = 1.8  # From https://www.ssi.dk/aktuelt/nyheder/2021/b117-kan-fore-til-flere-indlaggelser and https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/961042/S1095_NERVTAG_update_note_on_B.1.1.7_severity_20210211.pdf
                self.strain_label = strain

            # Known parameters on South African variant
            elif strain in choices['b1351']:
                strain_pars = dict()
                strain_pars['rel_beta'] = 1.4
                strain_pars['rel_severe_prob'] = 1.4
                strain_pars['rel_death_prob'] = 1.4
                strain_pars['rel_imm'] = 0.5
                self.strain_label = strain

            # Known parameters on Brazil variant
            elif strain in choices['p1']:
                strain_pars = dict()
                strain_pars['rel_beta'] = 1.4
                strain_pars['rel_severe_prob'] = 1.7
                strain_pars['rel_death_prob'] = 2.5
                strain_pars['dur'] = dict()
                strain_pars['dur']['crit2die'] = dict(dist='lognormal_int', par1=3, par2=2)
                strain_pars['rel_imm'] = 0.5
                self.strain_label = strain

            else:
                choicestr = '\n'.join(choices.values())
                errormsg = f'The selected variant "{strain}" is not implemented; choices are: {choicestr}'
                raise NotImplementedError(errormsg)

        # Option 2: strains can be specified as a dict of pars
        elif isinstance(strain, dict):
            strain_pars = strain
            self.strain_label = strain_label

        else:
            errormsg = f'Could not understand {type(strain)}, please specify as a string indexing a predefined strain or a dict.'
            raise ValueError(errormsg)

        return strain_pars

    def initialize(self, sim):
        if not hasattr(self, 'rel_imm'):
            self.rel_imm = 1

        # Update strain info
        for strain_key in cvd.strain_pars:
            if hasattr(self, strain_key):
                newval = getattr(self, strain_key)
                if strain_key == 'dur':  # Validate durations (make sure there are values for all durations)
                    newval = sc.mergenested(sim[strain_key][0], newval)
                sim[strain_key].append(newval)
            else:
                # use default
                print(f'{strain_key} not provided for this strain, using default value')
                sim[strain_key].append(sim[strain_key][0])

        self.initialized = True

    def apply(self, sim):

        if sim.t == self.days:  # Time to introduce strain

            # Check number of strains
            prev_strains = sim['n_strains']
            sim['n_strains'] += 1

            # Update strain-specific people attributes
            cvu.update_strain_attributes(sim.people) # don't think we need to do this if we just create people arrays with number of total strains in sim
            susceptible_inds = cvu.true(sim.people.susceptible)
            importation_inds = np.random.choice(susceptible_inds, self.n_imports)
            sim.people.infect(inds=importation_inds, layer='importation', strain=prev_strains)

        return


class Vaccine():
    '''
        Add a new vaccine to the sim (called by interventions.py vaccinate()

        stores number of doses for vaccine and a dictionary to pass to init_immunity for each dose

        Args:
            vaccine (dict or str): dictionary of parameters specifying information about the vaccine or label for loading pre-defined vaccine
            kwargs (dict):

        **Example**::
            moderna    = cv.Vaccine('moderna') # Create Moderna vaccine
            pfizer     = cv.Vaccine('pfizer) # Create Pfizer vaccine
            j&j        = cv.Vaccine('j&j') # Create J&J vaccine
            az         = cv.Vaccine('az) # Create AstraZeneca vaccine
            interventions += [cv.vaccinate(vaccines=[moderna, pfizer, j&j, az], days=[1, 10, 10, 30])] # Add them all to the sim
            sim = cv.Sim(interventions=interventions)
        '''

    def __init__(self, vaccine=None):

        self.rel_imm = None # list of length total_strains with relative immunity factor
        self.doses = None
        self.interval = None
        self.NAb_init = None
        self.NAb_boost = None
        self.vaccine_strain_info = self.init_strain_vaccine_info()
        self.vaccine_pars = self.parse_vaccine_pars(vaccine=vaccine)
        for par, val in self.vaccine_pars.items():
            setattr(self, par, val)
        return

    def init_strain_vaccine_info(self):
        # TODO-- populate this with data!
        rel_imm = {}
        rel_imm['known_vaccines'] = ['pfizer', 'moderna', 'az', 'j&j']
        rel_imm['known_strains'] = ['wild', 'b117', 'b1351', 'p1']
        for vx in rel_imm['known_vaccines']:
            rel_imm[vx] = {}
            rel_imm[vx]['wild'] = 1

        rel_imm['pfizer']['b117'] = 1/2
        rel_imm['pfizer']['b1351'] = 1/6.7
        rel_imm['pfizer']['p1'] = 1/6.5

        rel_imm['moderna']['b117'] = 1/1.8
        rel_imm['moderna']['b1351'] = 1/4.5
        rel_imm['moderna']['p1'] = 1/8.6

        rel_imm['az']['b1351'] = .5
        rel_imm['az']['p1'] = .5

        rel_imm['j&j']['b1351'] = .5
        rel_imm['j&j']['p1'] = .5

        return rel_imm

    def parse_vaccine_pars(self, vaccine=None):
        ''' Unpack vaccine information, which may be given in different ways'''

        # Option 1: vaccines can be chosen from a list of pre-defined strains
        if isinstance(vaccine, str):

            # List of choices currently available: new ones can be added to the list along with their aliases
            choices = {
                'pfizer': ['pfizer', 'Pfizer', 'Pfizer-BionTech'],
                'moderna': ['moderna', 'Moderna'],
                'az': ['az', 'AstraZeneca', 'astrazeneca'],
                'j&j': ['j&j', 'johnson & johnson', 'Johnson & Johnson'],
            }

            # (TODO: link to actual evidence)
            # Known parameters on pfizer
            if vaccine in choices['pfizer']:
                vaccine_pars = dict()
                vaccine_pars['NAb_init'] = dict(dist='normal', par1=0.5, par2= 2)
                vaccine_pars['doses'] = 2
                vaccine_pars['interval'] = 22
                vaccine_pars['NAb_boost'] = 3
                vaccine_pars['label'] = vaccine

            # Known parameters on moderna
            elif vaccine in choices['moderna']:
                vaccine_pars = dict()
                vaccine_pars['NAb_init'] = dict(dist='normal', par1=0.5, par2=2)
                vaccine_pars['doses'] = 2
                vaccine_pars['interval'] = 29
                vaccine_pars['NAb_boost'] = 3
                vaccine_pars['label'] = vaccine

            # Known parameters on az
            elif vaccine in choices['az']:
                vaccine_pars = dict()
                vaccine_pars['NAb_init'] = dict(dist='normal', par1=0.5, par2=2)
                vaccine_pars['doses'] = 2
                vaccine_pars['interval'] = 22
                vaccine_pars['NAb_boost'] = 3
                vaccine_pars['label'] = vaccine

            # Known parameters on j&j
            elif vaccine in choices['j&j']:
                vaccine_pars = dict()
                vaccine_pars['NAb_init'] = dict(dist='normal', par1=0.5, par2=2)
                vaccine_pars['doses'] = 1
                vaccine_pars['interval'] = None
                vaccine_pars['NAb_boost'] = 3
                vaccine_pars['label'] = vaccine

            else:
                choicestr = '\n'.join(choices.values())
                errormsg = f'The selected vaccine "{vaccine}" is not implemented; choices are: {choicestr}'
                raise NotImplementedError(errormsg)

        # Option 2: strains can be specified as a dict of pars
        elif isinstance(vaccine, dict):
            vaccine_pars = vaccine

        else:
            errormsg = f'Could not understand {type(vaccine)}, please specify as a string indexing a predefined vaccine or a dict.'
            raise ValueError(errormsg)

        return vaccine_pars

    def initialize(self, sim):

        ts = sim['total_strains']
        circulating_strains = ['wild'] # assume wild is circulating
        for strain in range(ts-1):
            circulating_strains.append(sim['strains'][strain].strain_label)

        if self.NAb_init is None :
            errormsg = f'Did not provide parameters for this vaccine'
            raise ValueError(errormsg)

        if self.rel_imm is None:
            print(f'Did not provide rel_imm parameters for this vaccine, trying to find values')
            self.rel_imm = []
            for strain in circulating_strains:
                if strain in self.vaccine_strain_info['known_strains']:
                    self.rel_imm.append(self.vaccine_strain_info[self.label][strain])
                else:
                    self.rel_imm.append(1)

        correct_size = len(self.rel_imm) == ts
        if not correct_size:
            errormsg = f'Did not provide relative immunity for each strain'
            raise ValueError(errormsg)

        return



# %% NAb methods
__all__ += ['init_nab', 'check_nab', 'nab_to_efficacy']

def init_nab(people, inds, prior_inf=True):
    '''
    Draws an initial NAb level for individuals.
    Can come from a natural infection or vaccination and depends on if there is prior immunity:
    1) a natural infection. If individual has no existing NAb, draw from distribution
    depending upon symptoms. If individual has existing NAb, multiply booster impact
    2) Vaccination. If individual has no existing NAb, draw from distribution
    depending upon vaccine source. If individual has existing NAb, multiply booster impact
    '''

    NAb_arrays = people.NAb[inds]
    prior_NAb_inds = cvu.idefined(NAb_arrays, inds) # Find people with prior NAbs
    no_prior_NAb_inds = np.setdiff1d(inds, prior_NAb_inds) # Find people without prior NAbs

    prior_NAb = people.NAb[prior_NAb_inds] # Array of NAb levels on this timestep for people with some NAbs
    peak_NAb = people.init_NAb[prior_NAb_inds]

    # NAbs from infection
    if prior_inf:
        NAb_boost = people.pars['NAb_boost']  # Boosting factor for natural infection
        # 1) No prior NAb: draw NAb from a distribution and compute
        if len(no_prior_NAb_inds):
            init_NAb = cvu.sample(**people.pars['NAb_init'], size=len(no_prior_NAb_inds))
            prior_symp = people.prior_symptoms[no_prior_NAb_inds]
            no_prior_NAb = (2**init_NAb) * prior_symp
            people.init_NAb[no_prior_NAb_inds] = no_prior_NAb

        # 2) Prior NAb: multiply existing NAb by boost factor
        if len(prior_NAb_inds):
            init_NAb = peak_NAb * NAb_boost
            people.init_NAb[prior_NAb_inds] = init_NAb

    # NAbs from a vaccine
    else:
        NAb_boost = people.pars['vaccine_info']['NAb_boost']  # Boosting factor for vaccination
        # 1) No prior NAb: draw NAb from a distribution and compute
        if len(no_prior_NAb_inds):
            init_NAb = cvu.sample(**people.pars['vaccine_info']['NAb_init'], size=len(no_prior_NAb_inds))
            people.init_NAb[no_prior_NAb_inds] = 2**init_NAb

        # 2) Prior NAb (from natural or vaccine dose 1): multiply existing NAb by boost factor
        if len(prior_NAb_inds):
            init_NAb = peak_NAb * NAb_boost
            people.NAb[prior_NAb_inds] = init_NAb

    return


def check_nab(t, people, inds=None):
    ''' Determines current NAbs based on date since recovered/vaccinated.'''

    # Indices of people who've had some NAb event
    rec_inds = cvu.defined(people.date_recovered[inds])
    vac_inds = cvu.defined(people.date_vaccinated[inds])
    both_inds = np.intersect1d(rec_inds, vac_inds)

    # Time since boost
    t_since_boost = np.full(len(inds), np.nan, dtype=cvd.default_int)
    t_since_boost[rec_inds] = t-people.date_recovered[inds[rec_inds]]
    t_since_boost[vac_inds] = t-people.date_vaccinated[inds[vac_inds]]
    t_since_boost[both_inds] = t-np.maximum(people.date_recovered[inds[both_inds]],people.date_vaccinated[inds[both_inds]])

    # Set current NAbs
    people.NAb[inds] = people.pars['NAb_kin'][t_since_boost] * people.init_NAb[inds]

    return


def nab_to_efficacy(nab, ax, slope, n_50):
    '''
    Convert NAb levels to immunity protection factors, using the functional form
    given in this paper: https://doi.org/10.1101/2021.03.09.21252641
    Inputs:
        nab (arr): an array of NAb levels
        ax (str): can be 'sus', 'symp' or 'sev', corresponding to the efficacy of protection against infection, symptoms, and severe disease respectively
    Returns:
        an array the same size as nab, containing the immunity protection factors for the specified axis
     '''

    if ax not in ['sus', 'symp', 'sev']:
        errormsg = f'Choice provided not in list of choices'
        raise ValueError(errormsg)

    n_50 = n_50[ax]
    # put in here nab to efficacy mapping (logistic regression from fig 1a from https://doi.org/10.1101/2021.03.09.21252641)
    efficacy = 1/(1+np.exp(-slope*(np.log10(nab) - np.log10(n_50)))) # from logistic regression in Khoury et al
    # efficacy = 1/(1+np.exp(-(slope*np.log10(2))*(nab - np.log2(n_50)))) # from logistic regression in Khoury et al
    return efficacy



# %% Immunity methods
__all__ += ['init_immunity', 'check_immunity']


def init_immunity(sim, create=False):
    ''' Initialize immunity matrices with all strains that will eventually be in the sim'''
    ts = sim['total_strains']
    immunity = {}

    # Pull out all of the circulating strains for cross-immunity
    circulating_strains = ['wild']
    for strain in sim['strains']:
        circulating_strains.append(strain.strain_label)

    # If immunity values have been provided, process them
    if sim['immunity'] is None or create:
        # Initialize immunity
        for ax in cvd.immunity_axes:
            if ax == 'sus':  # Susceptibility matrix is of size sim['n_strains']*sim['n_strains']
                immunity[ax] = np.full((ts, ts), sim['cross_immunity'],
                                       dtype=cvd.default_float)  # Default for off-diagnonals
                np.fill_diagonal(immunity[ax], 1)  # Default for own-immunity
            else:  # Progression and transmission are matrices of scalars of size sim['n_strains']
                immunity[ax] = np.full(ts, 1, dtype=cvd.default_float)

        known_strains = ['wild', 'b117', 'b1351', 'p1']
        cross_immunity = create_cross_immunity(circulating_strains)
        for i in range(ts):
            for j in range(ts):
                if i != j:
                    if circulating_strains[i] in known_strains and circulating_strains[j] in known_strains:
                        immunity['sus'][j][i] = cross_immunity[circulating_strains[j]][circulating_strains[i]]
        sim['immunity'] = immunity

    else:
        # if we know all the circulating strains, then update, otherwise use defaults
        known_strains = ['wild', 'b117', 'b1351', 'p1']
        cross_immunity = create_cross_immunity(circulating_strains)
        if sc.checktype(sim['immunity']['sus'], 'arraylike'):
            correct_size = sim['immunity']['sus'].shape == (ts, ts)
            if not correct_size:
                errormsg = f'Wrong dimensions for immunity["sus"]: you provided a matrix sized {sim["immunity"]["sus"].shape}, but it should be sized {(ts, ts)}'
                raise ValueError(errormsg)
            for i in range(ts):
                for j in range(ts):
                    if i != j:
                        if circulating_strains[i] in known_strains and circulating_strains[j] in known_strains:
                            sim['immunity']['sus'][j][i] = cross_immunity[circulating_strains[j]][
                                circulating_strains[i]]

        elif sc.checktype(sim['immunity']['sus'], dict):
            raise NotImplementedError
        else:
            errormsg = f'Type of immunity["sus"] not understood: you provided {type(sim["immunity"]["sus"])}, but it should be an array or dict.'
            raise ValueError(errormsg)

    # Next, precompute the NAb kinetics and store these for access during the sim
    sim['NAb_kin'] = pre_compute_waning(length=sim['n_days'], form=sim['NAb_decay']['form'], pars=sim['NAb_decay']['pars'])

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
    was_inf = cvu.true(people.t >= people.date_recovered)  # Had a previous exposure, now recovered
    is_vacc = cvu.true(people.vaccinated)  # Vaccinated
    date_rec = people.date_recovered  # Date recovered
    immunity = people.pars['immunity'] # cross-immunity/own-immunity scalars to be applied to NAb level before computing efficacy
    nab_eff_pars = people.pars['NAb_eff']

    # If vaccines are present, extract relevant information about them
    vacc_present = len(is_vacc)
    if vacc_present:
        vacc_info = people.pars['vaccine_info']

    if sus:
        ### PART 1:
        #   Immunity to infection for susceptible individuals
        is_sus = cvu.true(people.susceptible)  # Currently susceptible
        was_inf_same = cvu.true((people.recovered_strain == strain) & (people.t >= date_rec))  # Had a previous exposure to the same strain, now recovered
        was_inf_diff = np.setdiff1d(was_inf, was_inf_same)  # Had a previous exposure to a different strain, now recovered
        is_sus_vacc = np.intersect1d(is_sus, is_vacc)  # Susceptible and vaccinated
        is_sus_was_inf_same = np.intersect1d(is_sus, was_inf_same)  # Susceptible and being challenged by the same strain
        is_sus_was_inf_diff = np.intersect1d(is_sus, was_inf_diff)  # Susceptible and being challenged by a different strain

        if len(is_sus_vacc):
            vaccine_source = cvd.default_int(people.vaccine_source[is_sus_vacc])
            vaccine_scale = vacc_info['rel_imm'][vaccine_source, strain]
            current_NAbs = people.NAb[is_sus_vacc]
            people.sus_imm[strain, is_sus_vacc] = nab_to_efficacy(current_NAbs * vaccine_scale, 'sus', **nab_eff_pars)

        if len(is_sus_was_inf_same):  # Immunity for susceptibles with prior exposure to this strain
            current_NAbs = people.NAb[is_sus_was_inf_same]
            people.sus_imm[strain, is_sus_was_inf_same] = nab_to_efficacy(current_NAbs * immunity['sus'][strain, strain], 'sus', **nab_eff_pars)

        if len(is_sus_was_inf_diff):  # Cross-immunity for susceptibles with prior exposure to a different strain
            prior_strains = people.recovered_strain[is_sus_was_inf_diff]
            prior_strains_unique = cvd.default_int(np.unique(prior_strains))
            for unique_strain in prior_strains_unique:
                unique_inds = is_sus_was_inf_diff[cvu.true(prior_strains == unique_strain)]
                current_NAbs = people.NAb[unique_inds]
                people.sus_imm[strain, unique_inds] = nab_to_efficacy(current_NAbs * immunity['sus'][strain, unique_strain], 'sus', **nab_eff_pars)

    else:
        ### PART 2:
        #   Immunity to disease for currently-infected people
        is_inf_vacc = np.intersect1d(inds, is_vacc)
        was_inf = np.intersect1d(inds, was_inf)

        if len(is_inf_vacc):  # Immunity for infected people who've been vaccinated
            vaccine_source = cvd.default_int(people.vaccine_source[is_inf_vacc])
            vaccine_scale = vacc_info['rel_imm'][vaccine_source, strain]
            current_NAbs = people.NAb[is_inf_vacc]
            people.symp_imm[strain, is_inf_vacc] = nab_to_efficacy(current_NAbs * vaccine_scale * immunity['symp'][strain], 'symp', **nab_eff_pars)
            people.sev_imm[strain, is_inf_vacc] = nab_to_efficacy(current_NAbs * vaccine_scale * immunity['sev'][strain], 'sev', **nab_eff_pars)

        if len(was_inf):  # Immunity for reinfected people
            current_NAbs = people.NAb[was_inf]
            people.symp_imm[strain, was_inf] = nab_to_efficacy(current_NAbs * immunity['symp'][strain], 'symp', **nab_eff_pars)
            people.sev_imm[strain, was_inf] = nab_to_efficacy(current_NAbs * immunity['sev'][strain], 'sev', **nab_eff_pars)

    return



# %% Methods for computing waning
__all__ += ['pre_compute_waning']

def pre_compute_waning(length, form='nab_decay', pars=None):
    '''
    Process functional form and parameters into values
    - 'nab_decay'       : specific decay function taken from https://doi.org/10.1101/2021.03.09.21252641
    - 'exp_decay'       : exponential decay. Parameters should be init_val and half_life (half_life can be None/nan)
    - 'logistic_decay'  : logistic decay (TODO fill in details)
    - 'linear'          : linear decay (TODO fill in details)
    - others TBC!

    Args:
        length (float): length of array to return, i.e., for how long waning is calculated
        form (str):   the functional form to use
        pars (dict): passed to individual immunity functions
    Returns:
        array of length 'length' of values
    '''

    choices = [
        'nab_decay', # Default if no form is provided
        'exp_decay',
        'logistic_decay',
        'linear_growth',
        'linear_decay'
    ]

    # Process inputs
    if form is None or form == 'nab_decay':
        output = nab_decay(length, **pars)

    elif form == 'exp_decay':
        if pars['half_life'] is None: pars['half_life'] = np.nan
        output = exp_decay(length, **pars)

    elif form == 'logistic_decay':
        output = logistic_decay(length, **pars)

    elif form == 'linear_growth':
        output = linear_growth(length, **pars)

    elif form == 'linear_decay':
        output = linear_decay(length, **pars)

    else:
        choicestr = '\n'.join(choices)
        errormsg = f'The selected functional form "{form}" is not implemented; choices are: {choicestr}'
        raise NotImplementedError(errormsg)

    return output


def nab_decay(length, init_decay_rate, init_decay_time, decay_decay_rate):
    '''
    Returns an array of length 'length' containing the evaluated NAb decay
    function at each point
    Uses exponential decay, with the rate of exponential decay also set to exponentilly decay (!) after init_decay_time days
    '''

    f1  = lambda t, init_decay_rate: np.exp(-t*init_decay_rate)
    f2  = lambda t, init_decay_rate, init_decay_time, decay_decay_rate: np.exp(-t*(init_decay_rate*np.exp(-(t-init_decay_time)*decay_decay_rate)))
    t   = np.arange(length, dtype=cvd.default_int)
    y1  = f1(cvu.true(t<init_decay_time), init_decay_rate)
    y2  = f2(cvu.true(t>init_decay_time), init_decay_rate, init_decay_time, decay_decay_rate)
    y   = np.concatenate([y1,y2])
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
        result = np.concatenate(growth, decay, axis=None)
    else:
        t = np.arange(length, dtype=cvd.default_int)
        result = init_val * np.exp(-decay_rate * t)
    return result


def logistic_decay(length, init_val, decay_rate, half_val, lower_asymp, delay=None):
    ''' Calculate logistic decay '''

    if delay is not None:
        t = np.arange(length - delay, dtype=cvd.default_int)
        growth = linear_growth(delay, init_val / delay)
        decay = (init_val + (lower_asymp - init_val) / (
                1 + (t / half_val) ** decay_rate))
        result = np.concatenate((growth, decay), axis=None)
    else:
        t = np.arange(length, dtype=cvd.default_int)
        result = (init_val + (lower_asymp - init_val) / (
                1 + (t / half_val) ** decay_rate))
    return result  # TODO: make this robust to /0 errors


def linear_decay(length, init_val, slope):
    ''' Calculate linear decay '''
    t = np.arange(length, dtype=cvd.default_int)
    result = init_val - slope*t
    if result < 0:
        result = 0
    return result


def linear_growth(length, slope):
    ''' Calculate linear growth '''
    t = np.arange(length, dtype=cvd.default_int)
    return (slope * t)


def create_cross_immunity(circulating_strains):
    known_strains = ['wild', 'b117', 'b1351', 'p1']
    known_cross_immunity = dict()
    known_cross_immunity['wild'] = {} # cross-immunity to wild
    known_cross_immunity['wild']['b117'] = .5
    known_cross_immunity['wild']['b1351'] = .5
    known_cross_immunity['wild']['p1'] = .5
    known_cross_immunity['b117'] = {} # cross-immunity to b117
    known_cross_immunity['b117']['wild'] = 1
    known_cross_immunity['b117']['b1351'] = 1
    known_cross_immunity['b117']['p1'] = 1
    known_cross_immunity['b1351'] = {} # cross-immunity to b1351
    known_cross_immunity['b1351']['wild'] = 0.1
    known_cross_immunity['b1351']['b117'] = 0.1
    known_cross_immunity['b1351']['p1'] = 0.1
    known_cross_immunity['p1'] = {} # cross-immunity to p1
    known_cross_immunity['p1']['wild'] = 0.2
    known_cross_immunity['p1']['b117'] = 0.2
    known_cross_immunity['p1']['b1351'] = 0.2

    cross_immunity = {}
    cs = len(circulating_strains)
    for i in range(cs):
        cross_immunity[circulating_strains[i]] = {}
        for j in range(cs):
            if circulating_strains[j] in known_strains:
                if i != j:
                    if circulating_strains[i] in known_strains:
                        cross_immunity[circulating_strains[i]][circulating_strains[j]] = \
                            known_cross_immunity[circulating_strains[i]][circulating_strains[j]]

    return cross_immunity
