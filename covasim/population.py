'''
Defines functions for making the population.
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import requirements as cvreq
from . import utils as cvu
from . import data as cvdata
from . import defaults as cvd
from . import parameters as cvpars
from . import people as cvppl
from collections import defaultdict


# Specify all externally visible functions this file defines
__all__ = ['make_people', 'make_randpop', 'make_random_contacts',
           'make_microstructured_contacts', 'make_hybrid_contacts',
           'make_synthpop']


def make_people(sim, save_pop=False, popfile=None, verbose=None, die=True, reset=False):
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
    if popfile is None:
        popfile = sim.popfile

    # Check which type of population to produce
    if pop_type == 'synthpops':
        if not cvreq.check_synthpops():
            errormsg = f'You have requested "{pop_type}" population, but synthpops is not available; please use random, clustered, or hybrid'
            if die:
                raise ValueError(errormsg)
            else:
                print(errormsg)
                pop_type = 'random'

    # Actually create the population
    if sim.popdict and not reset:
        popdict = sim.popdict # Use stored one
        layer_keys = list(popdict['contacts'][0].keys()) # Assume there's at least one contact!
        sim.popdict = None # Once loaded, remove
    else:
        # Create the population
        if pop_type in ['random', 'clustered', 'hybrid']:
            popdict, layer_keys = make_randpop(sim, microstructure=pop_type)
        elif pop_type == 'synthpops':
            popdict, layer_keys = make_synthpop(sim)
        else:
            errormsg = f'Population type "{pop_type}" not found; choices are random, clustered, hybrid, or synthpops'
            raise NotImplementedError(errormsg)

    # Ensure prognoses are set
    if sim['prognoses'] is None:
        sim['prognoses'] = cvpars.get_prognoses(sim['prog_by_age'])

    # Actually create the people
    sim.layer_keys = layer_keys
    people = cvppl.People(sim.pars, **popdict) # List for storing the people
    sim.people = people

    average_age = sum(popdict['age']/pop_size)
    sc.printv(f'Created {pop_size} people, average age {average_age:0.2f} years', 2, verbose)

    if save_pop:
        if popfile is None:
            errormsg = 'Please specify a file to save to using the popfile kwarg'
            raise FileNotFoundError(errormsg)
        else:
            filepath = sc.makefilepath(filename=popfile)
            sc.saveobj(filepath, popdict)
            if verbose:
                print(f'Saved population of type "{pop_type}" with {pop_size:n} people to {filepath}')

    return


def make_randpop(sim, age_data=None, sex_ratio=0.5, microstructure=False):
    ''' Make a random population, without contacts '''

    pop_size = int(sim['pop_size']) # Number of people

    # Load age data based on 2018 Seattle demographics by default
    location = sim['location']
    if location is not None:
        try:
            age_data = cvdata.loaders.get_age_distribution(location)
        except ValueError as E:
            print(f'Could not load data for requested location "{location}" ({str(E)}), using default')
    if age_data is None:
        age_data = cvd.default_age_data

    # Handle sexes and ages
    uids = np.arange(pop_size, dtype=cvd.default_int)
    sexes = np.random.binomial(1, sex_ratio, pop_size)
    age_data_min  = age_data[:,0]
    age_data_max  = age_data[:,1] + 1 # Since actually e.g. 69.999
    age_data_range = age_data_max - age_data_min
    age_data_prob = age_data[:,2]
    age_data_prob /= age_data_prob.sum() # Ensure it sums to 1
    age_bins = cvu.multinomial(np.array(age_data_prob, dtype=cvd.default_float), cvd.default_int(pop_size)) # Choose age bins
    ages = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(pop_size) # Uniformly distribute within this age bin

    # Store output; data duplicated as per-person and list-like formats for convenience
    popdict = {}
    popdict['uid'] = uids
    popdict['age'] = ages
    popdict['sex'] = sexes

    if microstructure == 'random':
        contacts, layer_keys = make_random_contacts(pop_size, sim['contacts'])
    elif microstructure == 'clustered':
        contacts, layer_keys = make_microstructured_contacts(pop_size, sim['contacts'])
    elif microstructure == 'hybrid':
        contacts, layer_keys = make_hybrid_contacts(pop_size, ages, sim['contacts'])
    else:
        errormsg = f'Microstructure type "{microstructure}" not found; choices are random, clustered, or hybrid'
        raise NotImplementedError(errormsg)

    popdict['contacts'] = contacts

    return popdict, layer_keys


def make_random_contacts(pop_size, contacts, overshoot=1.2):
    ''' Make random static contacts '''

    # Preprocessing
    pop_size = int(pop_size) # Number of people
    contacts = sc.dcp(contacts)
    layer_keys = list(contacts.keys())
    contacts_list = []

    # Precalculate contacts
    n_across_layers = np.prod(list(contacts.values()))
    n_all_contacts  = int(pop_size*n_across_layers*overshoot)
    all_contacts    = cvu.choose_r(max_n=pop_size, n=n_all_contacts) # Choose people at random
    p_counts = {}
    for lkey in layer_keys:
        p_counts[lkey] = cvu.n_poisson(contacts[lkey], pop_size)  # Draw the number of Poisson contacts for this person

    # Make contacts
    count = 0
    for p in range(pop_size):
        contact_dict = {}
        for lkey in layer_keys:
            n_contacts = p_counts[lkey][p]
            contact_dict[lkey] = all_contacts[count:count+n_contacts] # Assign people
            count += n_contacts
        contacts_list.append(contact_dict)

    return contacts_list, layer_keys


def make_microstructured_contacts(pop_size, contacts):
    ''' Create microstructured contacts -- i.e. households, schools, etc. '''

    # Preprocessing -- same as above
    pop_size = int(pop_size) # Number of people
    contacts = sc.dcp(contacts)
    contacts.pop('c', None) # Remove community
    layer_keys = list(contacts.keys())
    contacts_list = [{c:[] for c in layer_keys} for p in range(pop_size)] # Pre-populate

    for layer_name, cluster_size in contacts.items():
        # Make clusters - each person belongs to one cluster
        n_remaining = pop_size
        contacts_dict = defaultdict(set) # Use defaultdict of sets for convenience while initializing. Could probably change this as part of performance optimization

        while n_remaining > 0:

            # Get the size of this cluster
            this_cluster =  cvu.poisson(cluster_size)  # Sample the cluster size
            if this_cluster > n_remaining:
                this_cluster = n_remaining

            # Indices of people in this cluster
            cluster_indices = (pop_size-n_remaining)+np.arange(this_cluster)

            # Add symmetric pairwise contacts in each cluster. Can probably optimize this
            for i in cluster_indices:
                for j in cluster_indices:
                    if j <= i:
                        pass
                    else:
                        contacts_dict[i].add(j)
                        contacts_dict[j].add(i)

            n_remaining -= this_cluster

        for key in contacts_dict.keys():
            contacts_list[key][layer_name] = np.array(list(contacts_dict[key]), dtype=cvd.default_int)

    return contacts_list, layer_keys


def make_hybrid_contacts(pop_size, ages, contacts, school_ages=None, work_ages=None):
    '''
    Create "hybrid" contacts -- microstructured contacts for households and
    random contacts for schools and workplaces, both of which have extremely
    basic age structure. A combination of both make_random_contacts() and
    make_microstructured_contacts().
    '''

    # Handle inputs and defaults
    layer_keys = ['h', 's', 'w', 'c']
    contacts = sc.mergedicts({'h':4, 's':20, 'w':20, 'c':20}, contacts) # Ensure essential keys are populated
    if school_ages is None:
        school_ages = [6, 18]
    if work_ages is None:
        work_ages   = [18, 65]

    # Create the empty contacts list -- a list of {'h':[], 's':[], 'w':[]}
    contacts_list = [{key:[] for key in layer_keys} for i in range(pop_size)]

    # Start with the household contacts for each person
    h_contacts, _ = make_microstructured_contacts(pop_size, {'h':contacts['h']})

    # Make community contacts
    c_contacts, _ = make_random_contacts(pop_size, {'c':contacts['c']})

    # Get the indices of people in each age bin
    ages = np.array(ages)
    s_inds = sc.findinds((ages >= school_ages[0]) * (ages < school_ages[1]))
    w_inds = sc.findinds((ages >= work_ages[0])   * (ages < work_ages[1]))

    # Create the school and work contacts for each person
    s_contacts, _ = make_random_contacts(len(s_inds), {'s':contacts['s']})
    w_contacts, _ = make_random_contacts(len(w_inds), {'w':contacts['w']})

    # Construct the actual lists of contacts
    for i     in range(pop_size):   contacts_list[i]['h']   =        h_contacts[i]['h']  # Copy over household contacts -- present for everyone
    for i,ind in enumerate(s_inds): contacts_list[ind]['s'] = s_inds[s_contacts[i]['s']] # Copy over school contacts
    for i,ind in enumerate(w_inds): contacts_list[ind]['w'] = w_inds[w_contacts[i]['w']] # Copy over work contacts
    for i     in range(pop_size):   contacts_list[i]['c']   =        c_contacts[i]['c']  # Copy over community contacts -- present for everyone

    return contacts_list, layer_keys



def make_synthpop(sim):
    ''' Make a population using synthpops, including contacts '''
    import synthpops as sp # Optional import
    pop_size = sim['pop_size']
    population = sp.make_population(n=pop_size)
    uids, ages, sexes, contacts = [], [], [], []
    for uid,person in population.items():
        uids.append(uid)
        ages.append(person['age'])
        sexes.append(person['sex'])

    # Replace contact UIDs with ints...
    uid_mapping = {uid:u for u,uid in enumerate(uids)}
    key_mapping = {'H':'h', 'S':'s', 'W':'w', 'C':'c'} # Remap keys from old names to new names
    for uid in uids:
        person = population.pop(uid)
        uid_contacts = sc.dcp(person['contacts'])
        int_contacts = {}
        for key in uid_contacts.keys():
            new_key = key_mapping[key]
            int_contacts[new_key] = []
            for uid in uid_contacts[key]:
                int_contacts[new_key].append(uid_mapping[uid])
            int_contacts[new_key] = np.array(int_contacts[new_key], dtype=cvd.default_int)
        contacts.append(int_contacts)

    # Add community contacts
    c_contacts, _ = make_random_contacts(pop_size, {'c':sim['contacts']['c']})
    for i in range(pop_size):
        contacts[i]['c'] = c_contacts[i]['c'] # Copy over community contacts -- present for everyone

    # Finalize
    popdict = {}
    popdict['uid']      = sc.dcp(uids)
    popdict['age']      = np.array(ages)
    popdict['sex']      = np.array(sexes)
    popdict['contacts'] = sc.dcp(contacts)
    layer_keys = list(key_mapping.values())

    return popdict, layer_keys
