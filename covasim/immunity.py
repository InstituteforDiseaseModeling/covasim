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

__all__ += ['Strain']


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
                'default': ['default', 'wild', 'pre-existing'],
                'b117': ['b117', 'B117', 'B.1.1.7', 'UK', 'uk', 'UK variant', 'uk variant'],
                'b1351': ['b1351', 'B1351', 'B.1.351', 'SA', 'sa', 'SA variant', 'sa variant'],
                # TODO: add other aliases
                'p1': ['p1', 'P1', 'P.1', 'Brazil', 'Brazil variant', 'brazil variant'],
            }

            # Empty pardict for wild strain
            if strain in choices['default']:
                strain_pars = dict()
                self.strain_label = strain

            # Known parameters on B117
            elif strain in choices['b117']:
                strain_pars = dict()
                strain_pars[
                    'rel_beta'] = 1.5  # Transmissibility estimates range from 40-80%, see https://cmmid.github.io/topics/covid19/uk-novel-variant.html, https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2020.26.1.2002106
                strain_pars[
                    'rel_severe_prob'] = 1.6  # From https://www.ssi.dk/aktuelt/nyheder/2021/b117-kan-fore-til-flere-indlaggelser and https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/961042/S1095_NERVTAG_update_note_on_B.1.1.7_severity_20210211.pdf
                self.strain_label = strain

            # Known parameters on South African variant
            elif strain in choices['b1351']:
                strain_pars = dict()
                strain_pars['imm_pars'] = dict()
                for ax in cvd.immunity_axes:
                    strain_pars['imm_pars'][ax] = dict(form='exp_decay', pars={'init_val': 1.,
                                                                               'half_life': 120})  # E484K mutation reduces immunity protection (TODO: link to actual evidence)
                self.strain_label = strain

            # Known parameters on Brazil variant
            elif strain in choices['p1']:
                strain_pars = dict()
                strain_pars['imm_pars'] = dict()
                for ax in cvd.immunity_axes:
                    strain_pars['imm_pars'][ax] = dict(form='exp_decay', pars={'init_val': 1.,
                                                                               'half_life': 120})  # E484K mutation reduces immunity protection (TODO: link to actual evidence)
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
        if not hasattr(self, 'imm_pars'):
            self.imm_pars = sim['imm_pars'][0]

        # Validate immunity pars (make sure there are values for all cvd.immunity_axes)
        for key in cvd.immunity_axes:
            if key not in self.imm_pars:
                print(f'Immunity pars for imported strain for {key} not provided, using default value')
                self.imm_pars[key] = sim['imm_pars'][0][key]

        # Update strain info
        for strain_key in cvd.strain_pars:
            if strain_key != 'immune_degree':
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
            # cvu.update_strain_attributes(sim.people) # don't think we need to do this if we just create people arrays with number of total strains in sim
            importation_inds = cvu.choose(max_n=len(sim.people),
                                          n=self.n_imports)  # TODO: do we need to check these people aren't infected? Or just consider it unlikely
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

        self.vaccine_immune_degree = None # dictionary of pre-loaded decay to by imm_axis and dose
        self.rel_imm = None # list of length total_strains with relative immunity factor
        self.doses = None
        self.imm_pars = None
        self.vaccine_pars = self.parse_vaccine_pars(vaccine=vaccine)
        for par, val in self.vaccine_pars.items():
            setattr(self, par, val)
        return

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
                vaccine_pars['imm_pars'] = {}
                for ax in cvd.immunity_axes:
                    vaccine_pars['imm_pars'][ax] = [dict(form='exp_decay', pars={'init_val': 0.5, 'half_life': 30}),
                                                    dict(form='exp_decay', pars={'init_val': 1., 'half_life': 180})]
                vaccine_pars['doses'] = 2

            # Known parameters on moderna
            elif vaccine in choices['moderna']:
                vaccine_pars = dict()
                vaccine_pars['imm_pars'] = {}
                for ax in cvd.immunity_axes:
                    vaccine_pars['imm_pars'][ax] = [dict(form='exp_decay', pars={'init_val': 0.5, 'half_life': 30}),
                                                    dict(form='exp_decay', pars={'init_val': 1., 'half_life': 180})]
                vaccine_pars['doses'] = 2

            # Known parameters on az
            elif vaccine in choices['az']:
                vaccine_pars = dict()
                vaccine_pars['imm_pars'] = {}
                for ax in cvd.immunity_axes:
                    vaccine_pars['imm_pars'][ax] = [dict(form='exp_decay', pars={'init_val': 0.5, 'half_life': 30}),
                                                    dict(form='exp_decay', pars={'init_val': 1., 'half_life': 180})]
                vaccine_pars['doses'] = 2

            # Known parameters on j&j
            elif vaccine in choices['j&j']:
                vaccine_pars = dict()
                vaccine_pars['imm_pars'] = {}
                for ax in cvd.immunity_axes:
                    vaccine_pars['imm_pars'][ax] = dict(form='exp_decay', pars={'init_val': 1., 'half_life': 120})
                vaccine_pars['doses'] = 1

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

        if self.imm_pars is None:
            errormsg = f'Did not provide parameters for this vaccine'
            raise ValueError(errormsg)

        if self.rel_imm is None:
            errormsg = f'Did not provide rel_imm parameters for this vaccine'
            raise ValueError(errormsg)

        correct_size = self.rel_imm.shape == ts
        if not correct_size:
            errormsg = f'Did not provide relative immunity for each strain'
            raise ValueError(errormsg)

        # Validate immunity pars (make sure there are values for all cvd.immunity_axes)
        for key in cvd.immunity_axes:
            if key not in self.imm_pars:
                errormsg = f'Immunity pars for vaccine for {key} not provided'
                raise ValueError(errormsg)

        ''' Initialize immune_degree with all strains that will eventually be in the sim'''
        doses = self.doses

        # Precompute waning
        immune_degree = []  # Stored as a list by dose
        for dose in range(doses):
            strain_immune_degree = {}
            for ax in cvd.immunity_axes:
                strain_immune_degree[ax] = pre_compute_waning(sim['n_days'], **self.imm_pars[ax][dose])
            immune_degree.append(strain_immune_degree)
        self.vaccine_immune_degree = immune_degree



# %% Immunity methods
__all__ += ['init_immunity', 'pre_compute_waning', 'init_vaccine_immunity']


def init_immunity(sim, create=False):
    ''' Initialize immunity matrices with all strains that will eventually be in the sim'''
    ts = sim['total_strains']
    immunity = {}

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
        sim['immunity'] = immunity

    else:
        if sc.checktype(sim['immunity']['sus'], 'arraylike'):
            correct_size = sim['immunity']['sus'].shape == (ts, ts)
            if not correct_size:
                errormsg = f'Wrong dimensions for immunity["sus"]: you provided a matrix sized {sim["immunity"]["sus"].shape}, but it should be sized {(ts, ts)}'
                raise ValueError(errormsg)
        elif sc.checktype(sim['immunity']['sus'], dict):
            # TODO: make it possible to specify this as something like:
            # imm = {'b117': {'wild': 0.4, 'p1': 0.3},
            #        'wild': {'b117': 0.6, 'p1': 0.7},
            #        'p1':   {'wild': 0.9, 'b117': 0.65}}
            # per Dan's suggestion, by using [s.strain_label for s in sim['strains']].
            # Would need lots of validation!!
            raise NotImplementedError
        else:
            errormsg = f'Type of immunity["sus"] not understood: you provided {type(sim["immunity"]["sus"])}, but it should be an array or dict.'
            raise ValueError(errormsg)

    # Precompute waning
    immune_degree = []  # Stored as a list by strain
    for s in range(ts):
        strain_immune_degree = {}
        for ax in cvd.immunity_axes:
            strain_immune_degree[ax] = pre_compute_waning(sim['n_days'], **sim['imm_pars'][s][ax])
        immune_degree.append(strain_immune_degree)
    sim['immune_degree'] = immune_degree


def pre_compute_waning(length, form, pars):
    '''
    Process immunity pars and functional form into a value
    - 'exp_decay'       : exponential decay (TODO fill in details)
    - 'logistic_decay'  : logistic decay (TODO fill in details)
    - 'linear'          : linear decay (TODO fill in details)
    - others TBC!

    Args:
        form (str):   the functional form to use
        pars (dict): passed to individual immunity functions
        length (float): length of time to compute immunity
    '''

    choices = [
        'exp_decay',
        'logistic_decay',
        'linear',
    ]

    # Process inputs
    if form == 'exp_decay':
        if pars['half_life'] is None: pars['half_life'] = np.nan
        output = exp_decay(length, **pars)

    elif form == 'logistic_decay':
        output = logistic_decay(length, **pars)

    else:
        choicestr = '\n'.join(choices)
        errormsg = f'The selected functional form "{form}" is not implemented; choices are: {choicestr}'
        raise NotImplementedError(errormsg)

    return output


# SPecific waning functions are listed here
def exp_decay(length, init_val, half_life):
    '''
    Returns an array of length t with values for the immunity at each time step after recovery
    '''
    decay_rate = np.log(2) / half_life if ~np.isnan(half_life) else 0.
    t = np.arange(length, dtype=cvd.default_int)
    return init_val * np.exp(-decay_rate * t)


def logistic_decay(length, init_val, decay_rate, half_val, lower_asymp):
    ''' Calculate logistic decay '''
    t = np.arange(length, dtype=cvd.default_int)
    return (init_val + (lower_asymp - init_val) / (
                1 + (t / half_val) ** decay_rate))  # TODO: make this robust to /0 errors
