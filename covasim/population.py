'''
Defines functions for making the population.
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import utils as cvu
from . import requirements as cvreqs
from . import person as cvper


# Specify all externally visible functions this file defines
__all__ = ['make_people', 'make_randpop', 'make_random_contacts', 'make_microstructured_contacts']



def make_people(sim, verbose=None, die=True, reset=False):
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
            popdict = make_synthpop()

    # Actually create the people
    people = [] # List for storing the people
    for p in range(n_people): # Loop over each person
        keys = ['uid', 'age', 'sex']
        person_args = {}
        for key in keys:
            person_args[key] = popdict[key][p] # Convert from list to dict
        person = cvper.Person(pars=sim.pars, **person_args) # Create the person
        people[person_args['uid']] = person # Save them to the dictionary

    # Store UIDs and people
    sim.popdict = popdict
    sim.people = people
    sim.contact_keys = list(sim['contacts'].keys())
    sim.contacts = popdict['contacts']

    average_age = sum(popdict['age']/n_people)
    sc.printv(f'Created {n_people} people, average age {average_age:0.2f} years', 1, verbose)

    return


def make_randpop(sim, age_data=None, sex_ratio=0.5):
    ''' Make a random population, without contacts '''

    n_people = int(sim['n']) # Number of people

    # Load age data based on 2018 Seattle demographics
    if age_data is None:
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

    # Handle sexes and ages
    sexes = cvu.rbt(sex_ratio, n_people)
    age_data_min  = age_data[:,0]
    age_data_max  = age_data[:,1] + 1 # Since actually e.g. 69.999
    age_data_range = age_data_max - age_data_min
    age_data_prob = age_data[:,2]
    age_data_prob /= age_data_prob.sum() # Ensure it sums to 1
    age_bins = cvu.mt(age_data_prob, n_people) # Choose age bins
    ages = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(n_people) # Uniformly distribute within this age bin

    # Store output; data duplicated as per-person and list-like formats for convenience
    popdict = {}
    popdict['age']      = ages
    popdict['sex']      = sexes
    popdict['contacts'] = make_random_contacts(sim)

    return popdict


def make_synthpop(sim):
    ''' Make a population using synthpops '''
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
    return popdict


def make_random_contacts(sim):
    # Make contacts
    n_people = int(sim['n']) # Number of people
    contacts = []
    for p in range(n_people):
        contact_dict = {'c':0}
        for key in sim['contacts'].keys():
            if key != 'c': # Skip community contacts, these are chosen afresh daily
                n_contacts = cvu.pt(sim['contacts'][key]) # Draw the number of Poisson contacts for this person
                contact_dict[key] = cvu.choose(max_n=n_people, n=n_contacts) # Choose people at random
        contacts.append(contact_dict)
    return contacts


def make_microstructured_contacts(sim):
    pass