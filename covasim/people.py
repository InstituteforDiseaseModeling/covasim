'''
Defines the Person class and functions associated with making people.
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import utils as cvu
from . import parameters as cvpars
from . import requirements as cvreqs


# Specify all externally visible functions this file defines
__all__ = ['Person', 'make_people', 'make_randpop', 'set_prognoses']


class Person(sc.prettyobj):
    '''
    Class for a single person.
    '''
    def __init__(self, pars, uid, age, sex, contacts, symp_prob, severe_prob, crit_prob, death_prob):
        self.uid         = str(uid) # This person's unique identifier
        self.age         = float(age) # Age of the person (in years)
        self.sex         = int(sex) # Female (0) or male (1)
        self.contacts    = contacts # The contacts this person has
        self.symp_prob   = symp_prob # Probability of developing symptoms
        self.severe_prob = severe_prob # Conditional probability of symptoms becoming severe, if symptomatic
        self.crit_prob   = crit_prob # Conditional probability of symptoms becoming critical, if severe
        self.death_prob  = death_prob # Conditional probability of dying, given severe symptoms
        self.OR_no_treat = pars['OR_no_treat']  # Increase in the probability of dying if treatment not available
        self.durpars     = pars['dur']  # Store duration parameters

        # Define state
        self.susceptible    = True
        self.exposed        = False
        self.infectious     = False
        self.symptomatic    = False
        self.severe         = False
        self.critical       = False
        self.diagnosed      = False
        self.recovered      = False
        self.dead           = False
        self.known_contact  = False # Keep track of whether each person is a contact of a known positive

        # Keep track of dates
        self.date_exposed      = None
        self.date_infectious   = None
        self.date_symptomatic  = None
        self.date_severe       = None
        self.date_critical     = None
        self.date_diagnosed    = None
        self.date_recovered    = None
        self.date_died         = None

        # Keep track of durations
        self.dur_exp2inf  = None # Duration from exposure to infectiousness
        self.dur_inf2sym  = None # Duration from infectiousness to symptoms
        self.dur_sym2sev  = None # Duration from symptoms to severe symptoms
        self.dur_sev2crit = None # Duration from symptoms to severe symptoms
        self.dur_disease  = None # Total duration of disease, from date of exposure to date of recovery or death

        self.infected = [] #: Record the UIDs of all people this person infected
        self.infected_by = None #: Store the UID of the person who caused the infection. If None but person is infected, then it was an externally seeded infection
        return


    def infect(self, t, bed_constraint=None, source=None):
        """
        Infect this person and determine their eventual outcomes.
            * Every infected person can infect other people, regardless of whether they develop symptoms
            * Infected people that develop symptoms are disaggregated into mild vs. severe (=requires hospitalization) vs. critical (=requires ICU)
            * Every asymptomatic, mildly symptomatic, and severely symptomatic person recovers
            * Critical cases either recover or die

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

        # Calculate how long before this person can infect other people
        self.dur_exp2inf     = cvu.sample(**self.durpars['exp2inf'])
        self.date_infectious = t + self.dur_exp2inf

        # Use prognosis probabilities to determine what happens to them
        symp_bool = cvu.bt(self.symp_prob) # Determine if they develop symptoms

        # CASE 1: Asymptomatic: may infect others, but have no symptoms and do not die
        if not symp_bool:  # No symptoms
            dur_asym2rec = cvu.sample(**self.durpars['asym2rec'])
            self.date_recovered = self.date_infectious + dur_asym2rec  # Date they recover
            self.dur_disease = self.dur_exp2inf + dur_asym2rec  # Store how long this person had COVID-19

        # CASE 2: Symptomatic: can either be mild, severe, or critical
        else:
            self.dur_inf2sym = cvu.sample(**self.durpars['inf2sym']) # Store how long this person took to develop symptoms
            self.date_symptomatic = self.date_infectious + self.dur_inf2sym # Date they become symptomatic
            sev_bool = cvu.bt(self.severe_prob) # See if they're a severe or mild case

            # CASE 2a: Mild symptoms, no hospitalization required and no probaility of death
            if not sev_bool: # Easiest outcome is that they're a mild case - set recovery date
                dur_mild2rec = cvu.sample(**self.durpars['mild2rec'])
                self.date_recovered = self.date_symptomatic + dur_mild2rec  # Date they recover
                self.dur_disease = self.dur_exp2inf + self.dur_inf2sym + dur_mild2rec  # Store how long this person had COVID-19

            # CASE 2b: Severe cases: hospitalization required, may become critical
            else:
                self.dur_sym2sev = cvu.sample(**self.durpars['sym2sev']) # Store how long this person took to develop severe symptoms
                self.date_severe = self.date_symptomatic + self.dur_sym2sev  # Date symptoms become severe
                crit_bool = cvu.bt(self.crit_prob)  # See if they're a critical case

                if not crit_bool:  # Not critical - they will recover
                    dur_sev2rec = cvu.sample(**self.durpars['sev2rec'])
                    self.date_recovered = self.date_severe + dur_sev2rec  # Date they recover
                    self.dur_disease = self.dur_exp2inf + self.dur_inf2sym + self.dur_sym2sev + dur_sev2rec  # Store how long this person had COVID-19

                # CASE 2c: Critical cases: ICU required, may die
                else:
                    self.dur_sev2crit = cvu.sample(**self.durpars['sev2crit'])
                    self.date_critical = self.date_severe + self.dur_sev2crit  # Date they become critical
                    this_death_prob = self.death_prob * (self.OR_no_treat if bed_constraint else 1.) # Probability they'll die
                    death_bool = cvu.bt(this_death_prob)  # Death outcome

                    if death_bool:
                        dur_crit2die = cvu.sample(**self.durpars['crit2die'])
                        self.date_died = self.date_critical + dur_crit2die # Date of death
                        self.dur_disease = self.dur_exp2inf + self.dur_inf2sym + self.dur_sym2sev + self.dur_sev2crit + dur_crit2die   # Store how long this person had COVID-19
                    else:
                        dur_crit2rec = cvu.sample(**self.durpars['crit2rec'])
                        self.date_recovered = self.date_critical + dur_crit2rec # Date they recover
                        self.dur_disease = self.dur_exp2inf + self.dur_inf2sym + self.dur_sym2sev + self.dur_sev2crit + dur_crit2rec  # Store how long this person had COVID-19

        if source:
            self.infected_by = source.uid
            source.infected.append(self.uid)

        return 1 # For incrementing counters


    def check_symptomatic(self, t):
        ''' Check for new progressions to symptomatic '''
        if not self.symptomatic and self.date_symptomatic and t >= self.date_symptomatic: # Person is changing to this state
            self.symptomatic = True
            return 1
        else:
            return 0


    def check_severe(self, t):
        ''' Check for new progressions to severe '''
        if not self.severe and self.date_severe and t >= self.date_severe: # Person is changing to this state
            self.severe = True
            return 1
        else:
            return 0


    def check_critical(self, t):
        ''' Check for new progressions to critical '''
        if not self.critical and self.date_critical and t >= self.date_critical: # Person is changing to this state
            self.critical = True
            return 1
        else:
            return 0


    def check_recovery(self, t):
        ''' Check if an infected person has recovered '''

        if not self.recovered and self.date_recovered and t >= self.date_recovered: # It's the day they recover
            self.exposed     = False
            self.infectious  = False
            self.symptomatic = False
            self.severe      = False
            self.critical    = False
            self.recovered   = True
            return 1
        else:
            return 0


    def check_death(self, t):
        ''' Check whether or not this person died on this timestep  '''
        if not self.dead and self.date_died and t >= self.date_died:
            self.exposed     = False
            self.infectious  = False
            self.symptomatic = False
            self.severe      = False
            self.critical    = False
            self.recovered   = False
            self.dead        = True
            return 1
        else:
            return 0


    def test(self, t, test_sensitivity):
        if self.infectious and cvu.bt(test_sensitivity):  # Person was tested and is true-positive
            self.diagnosed = True
            self.date_diagnosed = t
            return 1
        else:
            return 0


def make_people(sim, verbose=None, id_len=None, die=True, reset=False):
    '''
    Make the actual people for the simulation.

    Args:
        sim (Sim): the simulation object
        verbose (bool): level of detail to print
        id_len (int): length of ID for each person (default: calculate required length based on the number of people)
        die (bool): whether or not to fail if synthetic populations are requested but not available
        reset (bool): whether to force population creation even if self.popdict exists

    Returns:
        None.
    '''

    # Set inputs
    n_people     = int(sim['n']) # Shorten
    usepopdata   = sim['usepopdata'] # Shorten
    use_rand_pop = (usepopdata == 'random') # Whether or not to use a random population (as opposed to synthpops)

    # Set defaults
    if verbose is None: verbose = sim['verbose']
    if id_len  is None: id_len  = int(np.log10(n_people)) + 2 # Dynamically generate based on the number of people required

    # Check which type of population to rpoduce
    if not use_rand_pop and not cvreqs.available['synthpops']:
        errormsg = f'You have requested "{usepopdata}" population, but synthpops is not available; please use "random"'
        if die:
            raise ValueError(errormsg)
        else:
            print(errormsg)
            usepopdata = 'random'

    # Actually create the population
    if sim.popdict and not reset:
        popdict = sim.popdict # Use stored one
    else:
        if use_rand_pop:
            popdict = make_randpop(sim)
        else:
            import synthpops as sp # Optional import
            population = sp.make_population(n=sim['n'])
            uids, ages, sexes, contacts = [], [], [], []
            for uid,person in population.items():
                uids.append(uid)
                ages.append(person['age'])
                sexes.append(person['sex'])

            # Replace contact UIDs with ints...
            uid_mapping = {uid:u for u,uid in enumerate(uids)}
            key_mapping = {'H':'h', 'S':'s', 'W':'w', 'R':'c'} # Remap keys from old names to new names
            for uid,person in population.items():
                uid_contacts = person['contacts']
                int_contacts = {}
                for key in uid_contacts.keys():
                    new_key = key_mapping[key]
                    int_contacts[new_key] = []
                    for uid in uid_contacts[key]:
                        int_contacts[new_key].append(uid_mapping[uid])
                    int_contacts[new_key] = np.array(int_contacts[new_key], dtype=int)
                contacts.append(int_contacts)

            popdict = {}
            popdict['uid']      = uids
            popdict['age']      = np.array(ages)
            popdict['sex']      = np.array(sexes)
            popdict['contacts'] = contacts

    # Set prognoses by modifying popdict in place
    set_prognoses(sim, popdict)

    # Actually create the people
    people = {} # Dictionary for storing the people -- use plain dict since faster than odict
    for p in range(n_people): # Loop over each person
        keys = ['uid', 'age', 'sex', 'contacts', 'symp_prob', 'severe_prob', 'crit_prob', 'death_prob']
        person_args = {}
        for key in keys:
            person_args[key] = popdict[key][p] # Convert from list to dict
        person = Person(pars=sim.pars, **person_args) # Create the person
        people[person_args['uid']] = person # Save them to the dictionary

    # Store UIDs and people
    sim.popdict = popdict
    sim.uids = popdict['uid'] # Duplication, but used in an innermost loop so make as efficient as possible
    sim.people = people
    sim.contact_keys = list(sim['contacts'].keys())

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
        contact_dict = {'c':0}
        for key in sim['contacts'].keys():
            if key != 'c': # Skip community contacts, these are chosen afresh daily
                n_contacts = cvu.pt(sim['contacts'][key]) # Draw the number of Poisson contacts for this person
                contact_dict[key] = cvu.choose(max_n=n_people, n=n_contacts) # Choose people at random
        contacts.append(contact_dict)

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

    prog_pars = cvpars.get_default_prognoses(by_age=by_age)

    # If not by age, same value for everyone
    if not by_age:

        prognoses.symp_prob   = sim['rel_symp_prob']   * prog_pars.symp_prob   * np.ones(n)
        prognoses.severe_prob = sim['rel_severe_prob'] * prog_pars.severe_prob * np.ones(n)
        prognoses.crit_prob   = sim['rel_crit_prob']   * prog_pars.crit_prob   * np.ones(n)
        prognoses.death_prob  = sim['rel_death_prob']  * prog_pars.death_prob  * np.ones(n)

    # Otherwise, calculate probabilities of symptoms, severe symptoms, and death by age
    else:
        # Conditional probabilities of severe symptoms (given symptomatic) and death (given severe symptoms)
        severe_if_sym   = np.array([sev/sym  if sym>0  and sev/sym>0  else 0 for (sev,sym)  in zip(prog_pars.severe_probs, prog_pars.symp_probs)]) # Conditional probabilty of developing severe symptoms, given symptomatic
        crit_if_severe  = np.array([crit/sev if sev>0  and crit/sev>0 else 0 for (crit,sev) in zip(prog_pars.crit_probs,   prog_pars.severe_probs)]) # Conditional probabilty of developing critical symptoms, given severe
        death_if_crit   = np.array([dth/crit if crit>0 and dth/crit>0 else 0 for (dth,crit) in zip(prog_pars.death_probs,  prog_pars.crit_probs)])  # Conditional probabilty of dying, given critical

        symp_probs     = sim['rel_symp_prob']   * prog_pars.symp_probs  # Overall probability of developing symptoms
        severe_if_sym  = sim['rel_severe_prob'] * severe_if_sym         # Overall probability of developing severe symptoms (https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf)
        crit_if_severe = sim['rel_crit_prob']   * crit_if_severe        # Overall probability of developing critical symptoms (derived from https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm)
        death_if_crit  = sim['rel_death_prob']  * death_if_crit         # Overall probability of dying (https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf)

        # Calculate prognosis for each person
        symp_prob, severe_prob, crit_prob, death_prob  = [],[],[],[]
        age_cutoffs = prog_pars.age_cutoffs
        for age in ages:
            # Figure out which probability applies to a person of the specified age
            ind = next((ind for ind, val in enumerate([True if age < cutoff else False for cutoff in age_cutoffs]) if val), -1)
            this_symp_prob   = symp_probs[ind]     # Probability of developing symptoms
            this_severe_prob = severe_if_sym[ind]  # Probability of developing severe symptoms
            this_crit_prob   = crit_if_severe[ind] # Probability of developing critical symptoms
            this_death_prob  = death_if_crit[ind]  # Probability of dying after developing critical symptoms
            symp_prob.append(this_symp_prob)
            severe_prob.append(this_severe_prob)
            crit_prob.append(this_crit_prob)
            death_prob.append(this_death_prob)

        # Return output
        prognoses.symp_prob   = symp_prob
        prognoses.severe_prob = severe_prob
        prognoses.crit_prob   = crit_prob
        prognoses.death_prob  = death_prob

    popdict.update(prognoses) # Add keys to popdict

    return
