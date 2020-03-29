'''
Defines the Person class and functions associated with making people.
'''

#%% Imports
import numba as nb
import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import utils as cvu


# Specify all externally visible functions this file defines
__all__ = ['Person', 'make_people', 'set_person_attrs', 'set_prognosis']


class Person(sc.prettyobj):
    '''
    Class for a single person.
    '''
    def __init__(self, age, sex, symp_prob, severe_prob, death_prob, pars, uid=None, id_len=8):
        if uid is None:
            uid = sc.uuid(length=id_len) # Unique identifier for this person
        self.uid = str(uid)
        self.age = float(age) # Age of the person (in years)
        self.sex = int(sex) # Female (0) or male (1)
        self.symp_prob = symp_prob # Probability of developing symptoms
        self.severe_prob = severe_prob # Conditional probability of symptoms becoming sever, if symptomatic
        self.death_prob = death_prob # Conditional probability of dying, given severe symptoms

        # Define state
        self.alive          = True
        self.susceptible    = True
        self.exposed        = False
        self.infectious     = False
        self.symptomatic    = False
        self.severe         = False
        self.diagnosed      = False
        self.recovered      = False
        self.dead           = False
        self.known_contact  = False # Keep track of whether each person is a contact of a known positive

        # Infection property distributions
        self.dist_serial = dict(dist='normal_int', par1=pars['serial'],    par2=pars['serial_std'])
        self.dist_incub  = dict(dist='normal_int', par1=pars['incub'],     par2=pars['incub_std'])
        self.dist_dur    = dict(dist='normal_int', par1=pars['dur'],       par2=pars['dur_std'])
        self.dist_death  = dict(dist='normal_int', par1=pars['timetodie'], par2=pars['timetodie_std'])

        # Keep track of dates
        self.date_exposed     = None
        self.date_infectious  = None
        self.date_symptomatic = None
        self.date_diagnosed   = None
        self.date_recovered   = None
        self.date_died        = None

        self.infected = [] #: Record the UIDs of all people this person infected
        self.infected_by = None #: Store the UID of the person who caused the infection. If None but person is infected, then it was an externally seeded infection
        return


    def infect(self, t, source=None):
        """
        Infect this person

        This routine infects the person
        Args:
            source: A Person instance. If None, then it was a seed infection
        Returns:
            1 (for incrementing counters)
        """
        self.susceptible    = False
        self.exposed        = True
        self.date_exposed   = t

        # Calculate how long before they can infect other people
        serial_dist          = cvu.sample(**self.dist_serial)
        self.date_infectious = t + serial_dist

        # Use prognosis probabilities to determine what happens to them
        sym_bool = cvu.bt(self.symp_prob)

        if sym_bool:  # They develop symptoms
            self.date_symptomatic = t + cvu.sample(**self.dist_incub) # Date they become symptomatic
            self.severe = cvu.bt(self.severe_prob) # See if they're a severe or mild case
            death_bool = cvu.bt(self.death_prob) # TODO: incorporate health system
            if death_bool and self.severe: # They die
                self.date_died = t + cvu.sample(**self.dist_death) # Date of death
            else: # They recover
                self.date_recovered = self.date_infectious + cvu.sample(**self.dist_dur) # Date they recover

        else: # They recover
            self.date_recovered = self.date_infectious + cvu.sample(**self.dist_dur) # Date they recover

        if source:
            self.infected_by = source.uid
            source.infected.append(self.uid)

        infected = 1  # For incrementing counters

        return infected


    def check_death(self, t):
        ''' Check whether or not this person died on this timestep -- let's hope not '''

        if self.date_died and t >= self.date_died:
            self.exposed     = False
            self.infectious  = False
            self.symptomatic = False
            self.recovered   = False
            self.died        = True
            death = 1
        else:
            death = 0

        return death


    def check_recovery(self, t):
        ''' Check if an infected person has recovered '''

        if self.date_recovered and t >= self.date_recovered: # It's the day they recover
            self.exposed     = False
            self.infectious  = False
            self.symptomatic = False
            self.recovered   = True
            recovery = 1
        else:
            recovery = 0

        return recovery


    def test(self, t, test_sensitivity):
        if self.infectious and cvu.bt(test_sensitivity):  # Person was tested and is true-positive
            self.diagnosed = True
            self.date_diagnosed = t
            diagnosed = 1
        else:
            diagnosed = 0
        return diagnosed



def make_people(sim, verbose=None, id_len=None):

    if verbose is None: verbose = sim['verbose']
    if id_len  is None: id_len  = 6

    # Create the people -- just placeholders if we're using actual data
    people = {} # Dictionary for storing the people -- use plain dict since faster than odict
    n_people = int(sim['n'])
    uids = sc.uuid(which='ascii', n=n_people, length=id_len)
    for p in range(n_people): # Loop over each person
        uid = uids[p]
        if sim['usepopdata'] != 'random':
            age, sex, symp_prob, severe_prob, death_prob= -1, -1, -1, -1, -1 # These get overwritten later
        else:
            age, sex, symp_prob, severe_prob, death_prob = set_person_attrs(by_age=sim['prog_by_age'],
                                                                            default_symp_prob=sim['default_symp_prob'],
                                                                            default_severe_prob=sim['default_severe_prob'],
                                                                            default_death_prob=sim['default_death_prob'],
                                                                            use_data=False)
        person = Person(age=age, sex=sex, symp_prob=symp_prob, severe_prob=severe_prob, death_prob=death_prob, uid=uid, pars=sim.pars) # Create the person
        people[uid] = person # Save them to the dictionary

    # Store UIDs and people
    sim.uids = uids
    sim.people = people

    # Make the contact matrix -- TODO: move into a separate function
    if sim['usepopdata'] == 'random':
        sc.printv(f'Creating contact matrix without data...', 2, verbose)
        for p in range(int(sim['n'])):
            person = sim.get_person(p)
            person.n_contacts = cvu.pt(sim['contacts']) # Draw the number of Poisson contacts for this person
            person.contact_inds = cvu.choose_people(max_ind=len(sim.people), n=person.n_contacts) # Choose people at random, assigning to household
    else:
        sc.printv(f'Creating contact matrix with data...', 2, verbose)
        import synthpops as sp

        sim.contact_keys = list(sim['contacts_pop'].keys())

        make_contacts_keys = ['use_age','use_sex','use_loc','use_social_layers']
        options_args = dict.fromkeys(make_contacts_keys, True)
        if sim['usepopdata'] == 'bayesian':
            bayesian_args = sc.dcp(options_args)
            bayesian_args['use_bayesian'] = True
            bayesian_args['use_usa'] = False
            popdict = sp.make_popdict(uids=sim.uids, use_bayesian=True)
            contactdict = sp.make_contacts(popdict, options_args=bayesian_args)
        elif sim['usepopdata'] == 'data':
            data_args = sc.dcp(options_args)
            data_args['use_bayesian'] = False
            data_args['use_usa'] = True
            popdict = sp.make_popdict(uids=sim.uids, use_bayesian=False)
            contactdict = sp.make_contacts(popdict, options_args=data_args)

        contactdict = sc.odict(contactdict)
        for p,uid,entry in contactdict.enumitems():
            person = sim.get_person(p)
            person.age = entry['age']
            person.sex = entry['sex']
            person.cfr = set_cfr(person.age, default_cfr=sim['default_cfr'], cfr_by_age=sim['cfr_by_age'])
            person.contact_inds = entry['contacts']

    sc.printv(f'Created {sim["n"]} people, average age {sum([person.age for person in sim.people.values()])/sim["n"]:0.2f} years', 1, verbose)

    return


@nb.njit()
def _get_norm_age(min_age, max_age, age_mean, age_std):
    norm_age = np.random.normal(age_mean, age_std)
    age = np.minimum(np.maximum(norm_age, min_age), max_age)
    return age


def set_person_attrs(min_age=0, max_age=99, age_mean=40, age_std=15, default_symp_prob=None, default_severe_prob=None,
                     default_death_prob=None, by_age=True, use_data=True):
    '''
    Set the attributes for an individual, including:
        * age
        * sex
        * prognosis (i.e., how likely they are to develop symptoms/develop severe symptoms/die, based on age)
    '''
    sex = np.random.randint(2) # Define female (0) or male (1) -- evenly distributed
    age = _get_norm_age(min_age, max_age, age_mean, age_std)

    # Get the prognosis for a person of this age
    symp_prob, severe_prob, death_prob = set_prognosis(age=age, default_symp_prob=default_symp_prob, default_severe_prob=default_severe_prob, default_death_prob=default_death_prob, by_age=by_age)

    return age, sex, symp_prob, severe_prob, death_prob


def set_prognosis(age=None, default_symp_prob=0.7, default_severe_prob=0.2, default_death_prob=0.02, by_age=True):
    '''
    Determine the prognosis of an infected person: probability of being aymptomatic, or if symptoms develop, probability
    of developing severe symptoms and dying, based on their age
    '''
    # Overall probabilities of symptoms, severe symptoms, and death
    age_cutoffs  = [10,      20,      30,      40,      50,      60,      70,      80,      100]
    symp_probs   = [0.50,    0.55,    0.65,    0.70,    0.75,    0.80,    0.85,    0.90,    0.95]    # Overall probability of developing symptoms
    severe_probs = [0.00100, 0.00100, 0.01100, 0.03400, 0.04300, 0.08200, 0.11800, 0.16600, 0.18400] # Overall probability of developing severe symptoms (https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf)
    death_probs  = [0.00002, 0.00006, 0.00030, 0.00080, 0.00150, 0.00600, 0.02200, 0.05100, 0.09300] # Overall probability of dying (https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf)

    # Conditional probabilities of severe symptoms (given symptomatic) and death (given severe symptoms)
    severe_if_sym   = [sev/sym if sym>0 and sev/sym>0 else 0 for (sev,sym) in zip(severe_probs,symp_probs)]   # Conditional probabilty of developing severe symptoms, given symptomatic
    death_if_severe = [d/s if s>0 and d/s>0 else 0 for (d,s) in zip(death_probs,severe_probs)]                # Conditional probabilty of dying, given severe symptoms

    # Process different options for age
    # Not supplied, use default
    if age is None or not by_age:
        symp_prob, severe_prob, death_prob = default_symp_prob, default_severe_prob, default_death_prob

    # Single number
    elif sc.isnumber(age):

        # Figure out which probability applies to a person of the specified age
        ind = next((ind for ind, val in enumerate([True if age < cutoff else False for cutoff in age_cutoffs]) if val), -1)
        symp_prob    = symp_probs[ind]    # Probability of developing symptoms
        severe_prob = severe_if_sym[ind] # Probability of developing severe symptoms
        death_prob  = death_if_severe[ind] # Probability of dying after developing severe symptoms

    # Listlike
    elif sc.checktype(age, 'listlike'):
        symp_prob, severe_prob, death_prob  = [],[],[]
        for a in age:
            this_symp_prob, this_severe_prob, this_death_prob = set_prognosis(age=age, default_symp_prob=default_symp_prob, default_severe_prob=default_severe_prob, default_death_prob=default_death_prob, by_age=by_age)
            symp_prob.append(this_symp_prob)
            severe_prob.append(this_severe_prob)
            death_prob.append(this_death_prob)

    else:
        raise TypeError(f"set_prognosis accepts a single age or list/aray of ages, not type {type(age)}")

    return symp_prob, severe_prob, death_prob