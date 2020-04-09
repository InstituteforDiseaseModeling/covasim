'''
Defines functions for making the population.
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import utils as cvu
from . import defaults as cvd
from . import requirements as cvreqs
from . import parameters as cvpars
from . import person as cvper


# Specify all externally visible functions this file defines
__all__ = ['People', 'make_people', 'make_randpop', 'make_random_contacts', 'make_microstructured_contacts']


class People(list):
    '''
    A tiny class to handle the display of a very large number of people, which is
    prohibitively slow to print to screen. This is really just a list, except with
    the repr of an odict, and with a keys() method. It also has shortcuts for "filtering
    in" (i.e., keeping people with a certain attribute) and "filtering out" (removing
    people with a certain attribute).
    '''

    def filter_in(self, attr):
        '''
        Filter in based on an attribute.

        Args:
            attr (str): The attribute to filter on.

        Example:
            susceptibles = sim.people.filter_in('susceptible')
        '''
        return filter(lambda person: getattr(person, attr), self)


    def filter_out(self, attr):
        '''
        Filter out based on an attribute.

        Args:
            attr (str): The attribute to filter on.

        Example:
            not_susceptibles = sim.people.filter_out('susceptible')
        '''
        return filter(lambda person: not getattr(person, attr), self)


    def extract(self, attr):
        '''
        Return a list of a given attribute for every person.

        Args:
            attr (str): The attribute to extract.

        Example:
            ages = sim.people.extract('age')
        '''
        return [getattr(person, attr) for person in self]


    def keys(self):
        ''' Convenience method to list the "keys" of the list '''
        return list(range(len(self)))


    def __repr__(self, *args, **kwargs):
        ''' Why this class exists: better repr than a regular list '''
        return sc.odict.__repr__(self) # Use the odict repr to skip large numbers of people


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

    # Set inputs and defaults
    pop_size = int(sim['pop_size']) # Shorten
    pop_type = sim['pop_type'] # Shorten
    if verbose is None:
        verbose = sim['verbose']

    # Check which type of population to produce
    if pop_type == 'synthpops' and not cvreqs.available['synthpops']:
        errormsg = f'You have requested "{pop_type}" population, but synthpops is not available; please use "random" or "microstructure"'
        if die:
            raise ValueError(errormsg)
        else:
            print(errormsg)
            pop_type = 'random'

    # Actually create the population
    if sim.popdict and not reset:
        popdict = sim.popdict # Use stored one
    else:
        if pop_type == 'random':
            popdict = make_randpop(sim)
        elif pop_type == 'synthpops':
            sim.update_pars(use_layers=True) # If we use synthpops, we param['use_layers'] should always be True
            popdict = make_synthpop(sim)
        else:
            raise NotImplementedError

    # Ensure prognoses are set
    if sim['prognoses'] is None:
        sim['prognoses'] = cvpars.get_prognoses(sim['prog_by_age'])

    # Actually create the people
    people = People() # List for storing the people
    for p in range(pop_size): # Loop over each person
        keys = ['uid', 'age', 'sex', 'contacts']
        person_args = {}
        for key in keys:
            person_args[key] = popdict[key][p] # Convert from list to dict
        person = cvper.Person(pars=sim.pars, **person_args) # Create the person
        people.append(person) # Save them to the dictionary

    # Store people
    sim.popdict = popdict
    sim.people = people
    sim.contact_keys = popdict['contact_keys']

    average_age = sum(popdict['age']/pop_size)
    sc.printv(f'Created {pop_size} people, average age {average_age:0.2f} years', 1, verbose)

    return


def make_randpop(sim, age_data=None, sex_ratio=0.5):
    ''' Make a random population, without contacts '''

    pop_size = int(sim['pop_size']) # Number of people

    # Load age data based on 2018 Seattle demographics
    if age_data is None:
        age_data = cvd.default_age_data

    # Handle sexes and ages
    sexes = cvu.rbt(sex_ratio, pop_size)
    age_data_min  = age_data[:,0]
    age_data_max  = age_data[:,1] + 1 # Since actually e.g. 69.999
    age_data_range = age_data_max - age_data_min
    age_data_prob = age_data[:,2]
    age_data_prob /= age_data_prob.sum() # Ensure it sums to 1
    age_bins = cvu.mt(age_data_prob, pop_size) # Choose age bins
    ages = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(pop_size) # Uniformly distribute within this age bin

    # Store output; data duplicated as per-person and list-like formats for convenience
    popdict = {}
    popdict['uid'] = np.arange(pop_size, dtype=int)
    popdict['age'] = ages
    popdict['sex'] = sexes

    contacts, contact_keys = make_random_contacts(sim)
    popdict['contacts'] = contacts
    popdict['contact_keys'] = contact_keys

    return popdict


def make_synthpop(sim):
    ''' Make a population using synthpops, including contacts '''
    import synthpops as sp # Optional import
    population = sp.make_population(n=sim['pop_size'])
    uids, ages, sexes, contacts = [], [], [], []
    for uid,person in population.items():
        uids.append(uid)
        ages.append(person['age'])
        sexes.append(person['sex'])

    # Replace contact UIDs with ints...
    uid_mapping = {uid:u for u,uid in enumerate(uids)}
    key_mapping = {'H':'h', 'S':'s', 'W':'w', 'C':'c'} # Remap keys from old names to new names
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
    popdict['contact_keys'] = list(key_mapping.values())
    return popdict


def make_random_contacts(sim):
    ''' Make random static contacts '''
    pop_size = int(sim['pop_size']) # Number of people
    contacts_list = []
    contacts = sc.dcp(sim['contacts'])
    contact_keys = list(contacts.keys())
    for p in range(pop_size):
        contact_dict = {}
        for key in contact_keys:
            if key != 'c': # Added later
                n_contacts = cvu.pt(contacts[key]) # Draw the number of Poisson contacts for this person
                contact_dict[key] = cvu.choose(max_n=pop_size, n=n_contacts) # Choose people at random
        contacts_list.append(contact_dict)
    return contacts_list, contact_keys


def make_microstructured_contacts(sim):
    pass