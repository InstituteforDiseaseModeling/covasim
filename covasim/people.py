'''
Defines the Person class and functions associated with making people.
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import utils as cvu
from . import requirements as cvreqs


# Specify all externally visible functions this file defines
__all__ = ['Person', 'make_people', 'make_randpop', 'set_prognoses']


class Person(sc.prettyobj):
    '''
    Class for a single person.
    '''
    def __init__(self, pars, uid, age, sex, contacts, symp_prob, severe_prob, death_prob):
        self.uid         = str(uid) # This person's unique identifier
        self.age         = float(age) # Age of the person (in years)
        self.sex         = int(sex) # Female (0) or male (1)
        self.contacts    = contacts # The contacts this person has
        self.symp_prob   = symp_prob # Probability of developing symptoms
        self.severe_prob = severe_prob # Conditional probability of symptoms becoming severe, if symptomatic
        self.death_prob  = death_prob # Conditional probability of dying, given severe symptoms
        self.OR_no_treat = pars['OR_no_treat']  # Increase in the probability of dying if treatment not available

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
        self.dist_serial  = dict(dist='normal_int', par1=pars['serial'],    par2=pars['serial_std'])
        self.dist_incub   = dict(dist='normal_int', par1=pars['incub'],     par2=pars['incub_std'])
        self.dist_sev     = dict(dist='normal_int', par1=pars['severe'],    par2=pars['severe_std'])
        self.dist_dur     = dict(dist='normal_int', par1=pars['dur'],       par2=pars['dur_std'])
        self.dist_dur_sev = dict(dist='normal_int', par1=pars['dur_sev'],   par2=pars['dur_sev_std'])
        self.dist_death   = dict(dist='normal_int', par1=pars['timetodie'], par2=pars['timetodie_std'])

        # Keep track of dates
        self.date_exposed      = None
        self.date_infectious   = None
        self.date_symptomatic  = None
        self.date_severe       = None
#        self.date_hospitalized = None
        self.date_diagnosed    = None
        self.date_recovered    = None
        self.date_died         = None

        self.infected = [] #: Record the UIDs of all people this person infected
        self.infected_by = None #: Store the UID of the person who caused the infection. If None but person is infected, then it was an externally seeded infection
        return


    def infect(self, t, bed_constraint=None, source=None):
        """
        Infect this person and determine their eventual outcomes.
            * Every infected person can infect other people, regardless of whether they develop symptoms
            * Infected people that develop symptoms are disaggregated into mild vs. severe (=requires hospitalization)
            * Every asymptomatic infected person recovers
            * Symptomatic infected people either recover or die, with the probability of recovery affected by whether they are hospitalized

        Args:
            t: (int) timestep
            bed_constraint: (bool) whether or not there is a bed available for this person
            source: (Person instance), if None, then it was a seed infection

        Returns:
            1 (for incrementing counters)
        """
        self.susceptible    = False
        self.exposed        = True
        self.date_exposed   = t

        # Deal with bed constraint if applicable
        if bed_constraint is None: bed_constraint = False

        # Calculate how long before they can infect other people
        serial_dist          = cvu.sample(**self.dist_serial)
        self.date_infectious = t + serial_dist

        # Use prognosis probabilities to determine what happens to them
        symp_bool = cvu.bt(self.symp_prob) # Determine if they develop symptoms

        # CASE 1: Asymptomatic: may infect others, but no symptoms and no probability of death
        if not symp_bool:  # No symptoms
            self.date_recovered = self.date_infectious + cvu.sample(**self.dist_dur)  # Date they recover

        # CASE 2: Symptomatic: can either be a mild case or a severe case
        else:
            self.date_symptomatic = t + cvu.sample(**self.dist_incub) # Date they become symptomatic
            sev_bool = cvu.bt(self.severe_prob) # See if they're a severe or mild case

            # CASE 2a: Mild symptoms, no hospitalization required and no probaility of death
            if not sev_bool: # Easiest outcome is that they're a mild case - set recovery date
                self.date_recovered = self.date_infectious + cvu.sample(**self.dist_dur)  # Date they recover

            # CASE 2b: Severe cases: hospitalization required, death possible
            else:
                self.date_severe = self.date_symptomatic + cvu.sample(**self.dist_sev)  # Date symptoms become severe
                this_death_prob = self.death_prob * (self.OR_no_treat if bed_constraint else 1.) # Probability they'll die
                death_bool = cvu.bt(this_death_prob)  # Death outcome
                #if not bed_constraint: self.date_hospitalized = self.date_severe  # They get hospitalized when symptoms become severe
                if death_bool: self.date_died = t + cvu.sample(**self.dist_death)  # Date of death
                else: self.date_recovered = self.date_severe + cvu.sample(**self.dist_dur_sev) # Date they recover

        if source:
            self.infected_by = source.uid
            source.infected.append(self.uid)

        infected = 1  # For incrementing counters

        return infected


    def check_death(self, t):
        ''' Check whether or not this person died on this timestep  '''
        if self.date_died and t == self.date_died:
            self.exposed     = False
            self.infectious  = False
            self.symptomatic = False
            self.severe      = False
            self.recovered   = False
            self.died        = True
            death = 1
        else:
            death = 0

        return death


    def check_symptomatic(self, t):
        ''' Check if an infected person has developed symptoms '''
        if self.date_symptomatic and t == self.date_symptomatic: # Person is symptomatic
            self.symptomatic = True
            symptomatic = 1
        else:
            symptomatic = 0
        return symptomatic


    def check_severe(self, t):
        ''' Check if an infected person has developed severe symptoms requiring hospitalization'''
        if self.date_severe and t == self.date_severe: # Symptoms have become bad enough to need hospitalization
            self.severe = True
            severe = 1
        else:
            severe = 0
        return severe


    def check_recovery(self, t):
        ''' Check if an infected person has recovered '''

        if self.date_recovered and t == self.date_recovered: # It's the day they recover
            self.exposed     = False
            self.infectious  = False
            self.symptomatic = False
            self.severe      = False
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



def make_people(sim, verbose=None, id_len=None, die=True):

    # Set defaults
    if verbose is None: verbose = sim['verbose']
    if id_len  is None: id_len  = 6

    # Set inputs
    n_people     = int(sim['n']) # Shorten
    usepopdata   = sim['usepopdata'] # Shorten
    use_rand_pop = (usepopdata == 'random') # Whether or not to use a random population (as opposed to synthpops)

    # Check which type of population to rpoduce
    if not use_rand_pop and not cvreqs.available['synthpops']:
        errormsg = f'You have requested "{usepopdata}" population, but synthpops is not available; please use "random"'
        if die:
            raise ValueError(errormsg)
        else:
            print(errormsg)
            usepopdata = 'random'

    # Actually create the population
    if use_rand_pop:
        popdict = make_randpop(sim)
    else:
        import synthpops as sp # Optional import
        popdict = sp.make_population(n=sim['n'])

    # Set prognoses by modifying popdict in place
    set_prognoses(sim, popdict)

    # Actually create the people
    people = {} # Dictionary for storing the people -- use plain dict since faster than odict
    for p in range(n_people): # Loop over each person
        keys = ['uid', 'age', 'sex', 'contacts', 'symp_prob', 'severe_prob', 'death_prob']
        person_args = {}
        for key in keys:
            person_args[key] = popdict[key][p] # Convert from list to dict
        person = Person(pars=sim.pars, **person_args) # Create the person
        people[person_args['uid']] = person # Save them to the dictionary

    # Store UIDs and people
    sim.uids = popdict['uid']
    sim.people = people
    sim.contact_keys = list(sim['contacts_pop'].keys())

    average_age = sum(popdict['age']/n_people)
    sc.printv(f'Created {n_people} people, average age {average_age:0.2f} years', 1, verbose)

    return

def make_randpop(sim, id_len=6):
    ''' Make a random population, without contacts '''

    # Load age data based on 2018 Seattle demographics
    age_data = np.array([
        [ 0,  4, 0.0605],
        [ 5,  9, 0.0607],
        [10, 14, 0.0566],
        [15, 19, 0.0557],
        [20, 24, 0.0612],
        [25, 29, 0.0843],
        [30, 34, 0.0848],
        [35, 39, 0.0764],
        [40, 44, 0.0697],
        [45, 49, 0.0701],
        [50, 54, 0.0681],
        [55, 59, 0.0653],
        [60, 64, 0.0591],
        [65, 69, 0.0453],
        [70, 74, 0.0312],
        [75, 79, 0.02016], # Calculated based on 0.0504 total for >=75
        [80, 84, 0.01344],
        [85, 89, 0.01008],
        [90, 99, 0.00672],
        ])

    # Handle sex and UID
    n_people = int(sim['n']) # Number of people
    uids = sc.uuid(which='ascii', n=n_people, length=id_len)
    sexes = cvu.rbt(0.5, n_people)

    # Handle ages
    age_data_min  = age_data[:,0]
    age_data_max  = age_data[:,1] + 1 # Since actually e.g. 69.999
    age_data_range = age_data_max - age_data_min
    age_data_prob = age_data[:,2]
    age_data_prob /= age_data_prob.sum() # Ensure it sums to 1
    age_bins = cvu.mt(age_data_prob, n_people) # Choose age bins
    ages = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(n_people) # Uniformly distribute within this age bin

    # Make contacts
    contacts = []
    for p in range(n_people):
        n_contacts = cvu.pt(sim['contacts']) # Draw the number of Poisson contacts for this person
        contact_inds = cvu.choose(max_n=n_people, n=n_contacts) # Choose people at random, assigning to household
        contacts.append(contact_inds)

    # Store output; data duplicated as per-person and list-like formats for convenience
    popdict = {}
    popdict['uid']     = uids
    popdict['age']     = ages
    popdict['sex']    = sexes
    popdict['contacts'] = contacts

    return popdict


def set_prognoses(sim, popdict):
    '''
    Determine the prognosis of an infected person: probability of being aymptomatic, or if symptoms develop, probability
    of developing severe symptoms and dying, based on their age
    '''

    # Initialize input and output
    by_age = sim['prog_by_age']
    ages = sc.promotetoarray(popdict['age']) # Ensure it's an array
    n = len(ages)
    prognoses = sc.objdict()

    # If not by age, same value for everyone
    if not by_age:
        prognoses.symp_prob    = sim['default_symp_prob']*np.ones(n)
        prognoses.severe_prob = sim['default_severe_prob']*np.ones(n)
        prognoses.death_prob  = sim['default_death_prob']*np.ones(n)

    else:
        # Overall probabilities of symptoms, severe symptoms, and death
        age_cutoffs  = [10,      20,      30,      40,      50,      60,      70,      80,      100]
        symp_probs   = [0.50,    0.55,    0.65,    0.70,    0.75,    0.80,    0.85,    0.90,    0.95]    # Overall probability of developing symptoms
        severe_probs = [0.00100, 0.00100, 0.01100, 0.03400, 0.04300, 0.08200, 0.11800, 0.16600, 0.18400] # Overall probability of developing severe symptoms (https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf)
        death_probs  = [0.00002, 0.00006, 0.00030, 0.00080, 0.00150, 0.00600, 0.02200, 0.05100, 0.09300] # Overall probability of dying (https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf)

        # Conditional probabilities of severe symptoms (given symptomatic) and death (given severe symptoms)
        severe_if_sym   = [sev/sym if sym>0 and sev/sym>0 else 0 for (sev,sym) in zip(severe_probs,symp_probs)]   # Conditional probabilty of developing severe symptoms, given symptomatic
        death_if_severe = [d/s if s>0 and d/s>0 else 0 for (d,s) in zip(death_probs,severe_probs)]                # Conditional probabilty of dying, given severe symptoms

        # Calculate prognosis for each person
        symp_prob, severe_prob, death_prob  = [],[],[]
        for age in ages:
            # Figure out which probability applies to a person of the specified age
            ind = next((ind for ind, val in enumerate([True if age < cutoff else False for cutoff in age_cutoffs]) if val), -1)
            this_symp_prob    = symp_probs[ind]    # Probability of developing symptoms
            this_severe_prob = severe_if_sym[ind] # Probability of developing severe symptoms
            this_death_prob  = death_if_severe[ind] # Probability of dying after developing severe symptoms
            symp_prob.append(this_symp_prob)
            severe_prob.append(this_severe_prob)
            death_prob.append(this_death_prob)

        # Return output
        prognoses.symp_prob    = symp_prob
        prognoses.severe_prob = severe_prob
        prognoses.death_prob  = death_prob

    popdict.update(prognoses) # Add keys to popdict

    return
