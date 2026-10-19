'''
Defines functions for making the population.
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import requirements as cvreq
from . import utils as cvu
from . import misc as cvm
from . import base as cvb
from . import data as cvdata
from . import defaults as cvd
from . import parameters as cvpar
from . import people as cvppl
from .geography import assign_density_based_coords
from .sewersheds import assign_sewersheds
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Specify all externally visible functions this file defines
__all__ = ['make_people', 'make_randpop', 'assign_regions',
           'assign_density_based_coords', 'make_random_contacts',
           'make_microstructured_contacts', 'make_hybrid_contacts',
           'make_synthpop', 'make_rank_based_contacts']

def load_population_regions(filename):
    regions = gpd.read_file(filename)

    required = {
        "region_code",
        "region_name",
        "population",
    }

    missing = required - set(regions.columns)

    if missing:
        raise ValueError(
            f"Population boundary file is missing columns: {missing}"
        )

    if regions.empty:
        raise ValueError("Population boundary file contains no regions.")

    regions["population"] = pd.to_numeric(
        regions["population"],
        errors="raise",
    )

    if (regions["population"] < 0).any():
        raise ValueError("Population cannot be negative.")

    total_population = regions["population"].sum()

    if total_population <= 0:
        raise ValueError("Total population must be greater than zero.")

    regions["probability"] = (
        regions["population"] / total_population
    )

    return regions

def assign_regions(pars, population_data):
    pop_size = int(pars["pop_size"])

    labels = population_data["region_code"].to_numpy()
    probabilities = population_data["probability"].to_numpy()

    return np.random.choice(
        labels,
        size=pop_size,
        p=probabilities,
    )

def assign_coords(assigned_region_codes, population_data):
    """
    Generate coordinates for agents based on their assigned region geometry.
    
    assigned_region_codes: array of region_code values assigned to each agent.
    population_data: GeoDataFrame loaded by load_population_regions.
    """
    # Index the population GeoDataFrame by region_code for fast lookup
    pop_indexed = population_data.set_index("region_code")

    xs = []
    ys = []

    for r_code in assigned_region_codes:
        if r_code not in pop_indexed.index:
            # Fallback if region code isn't found
            xs.append(0.0)
            ys.append(0.0)
            continue

        geom = pop_indexed.loc[r_code, "geometry"]
        minx, miny, maxx, maxy = geom.bounds

        # Uniform rejection sampling inside the assigned polygon
        # (Guarantees coordinates fall strictly inside the region shape)
        while True:
            rx = np.random.uniform(minx, maxx)
            ry = np.random.uniform(miny, maxy)
            if geom.contains(Point(rx, ry)):
                xs.append(rx)
                ys.append(ry)
                break

    return np.column_stack((xs, ys))

def make_people(sim, popdict=None, die=True, reset=False, recreate=False, verbose=None, **kwargs):
    '''
    Make the actual people for the simulation.

    Usually called via ``sim.initialize()``. While in theory this function can be
    called directly by the user, usually it's better to call ``cv.People()`` directly.

    Args:
        sim      (Sim)  : the simulation object; population parameters are taken from the sim object
        popdict  (any)  : if supplied, use this population dictionary instead of generating a new one; can be a dict, SynthPop, or People object
        die      (bool) : whether or not to fail if synthetic populations are requested but not available
        reset    (bool) : whether to force population creation even if self.popdict/self.people exists
        recreate (bool) : whether to recreate (re-instantiate) the People object even if already supplied
        verbose  (bool) : level of detail to print
        kwargs   (dict) : passed to make_randpop() or make_synthpop()

    Returns:
        people (People): people
    '''

    # Set inputs and defaults
    pop_size = int(sim['pop_size']) # Shorten
    pop_type = sim['pop_type'] # Shorten
    if verbose is None:
        verbose = sim['verbose']

    # Check which type of population to produce
    if pop_type == 'synthpops':
        if not cvreq.check_synthpops(): # pragma: no cover
            errormsg = f'You have requested "{pop_type}" population, but synthpops is not available; please use random, clustered, or hybrid'
            if die:
                raise ValueError(errormsg)
            else:
                print(errormsg)
                pop_type = 'hybrid'

        location = sim['location']
        if location and verbose: # pragma: no cover
            warnmsg = f'Not setting ages or contacts for "{location}" since synthpops contacts are pre-generated'
            cvm.warn(warnmsg)

    # If a people object or popdict is supplied, use it
    if sim.people and not reset:
        assign_sewersheds(sim.people, sim.pars)
        sim.people.initialize(sim_pars=sim.pars)
        return sim.people # If it's already there, just return
    elif sim.popdict and popdict is None:
        popdict = sim.popdict # Use stored one
        sim.popdict = None # Once loaded, remove

    # Handle SynthPops separately: run the popdict through the function even if it already exists
    if pop_type == 'synthpops':
        popdict = make_synthpop(sim, popdict=popdict, **kwargs)

    # Main use case: no popdict is supplied, so create one
    else:
        if popdict is None:
            if pop_type in ['random', 'hybrid', 'rank']:
                popdict = make_randpop(sim, microstructure=pop_type, **kwargs) # Main use case: create a random or hybrid or rank-based population
            else: # pragma: no cover
                errormsg = f'Population type "{pop_type}" not found; choices are random, hybrid, or synthpops'
                raise ValueError(errormsg)

    # Ensure prognoses are set
    if sim['prognoses'] is None:
        sim['prognoses'] = cvpar.get_prognoses(sim['prog_by_age'], version=sim._default_ver)

    # Do minimal validation and create the people
    validate_popdict(popdict, sim.pars, verbose=verbose)
    if isinstance(popdict, cvppl.People) and not recreate:
        people = popdict
        people.set_pars(sim.pars)
    else:
        spatial = {key: popdict[key] for key in ('x', 'y') if key in popdict.keys()}
        people = cvppl.People(sim.pars, uid=popdict['uid'], age=popdict['age'], sex=popdict['sex'], contacts=popdict['contacts'], region=popdict['region'], **spatial) # List for storing the people

    assign_sewersheds(people, sim.pars)

    sc.printv(f'Created {pop_size} people, average age {people.age.mean():0.2f} years', 2, verbose)

    return people


def validate_popdict(popdict, pars, verbose=True):
    '''
    Check that the popdict is the correct type, has the correct keys, and has
    the correct length
    '''

    # Check it's the right type
    try:
        popdict.keys() # Although not used directly, this is used in the error message below, and is a good proxy for a dict-like object
    except Exception as E:
        errormsg = f'The popdict should be a dictionary or cv.People object, but instead is {type(popdict)}'
        raise TypeError(errormsg) from E

    # Check keys and lengths
    required_keys = ['uid', 'age', 'sex']
    popdict_keys = popdict.keys()
    pop_size = pars['pop_size']
    for key in required_keys:

        if key not in popdict_keys:
            errormsg = f'Could not find required key "{key}" in popdict; available keys are: {sc.strjoin(popdict.keys())}'
            sc.KeyNotFoundError(errormsg)

        actual_size = len(popdict[key])
        if actual_size != pop_size:
            errormsg = f'Could not use supplied popdict since key {key} has length {actual_size}, but all keys must have length {pop_size}'
            raise ValueError(errormsg)

        isnan = np.isnan(popdict[key]).sum()
        if isnan:
            errormsg = f'Population not fully created: {isnan:,} NaNs found in {key}. This can be caused by calling cv.People() instead of cv.make_people().'
            raise ValueError(errormsg)

    if ('contacts' not in popdict_keys) and (not hasattr(popdict, 'contacts')) and verbose:
        warnmsg = 'No contacts found. Please remember to add contacts before running the simulation.'
        cvm.warn(warnmsg)

    return


def make_randpop(pars, use_age_data=True, use_household_data=True, sex_ratio=0.5, microstructure='random', **kwargs):
    '''
    Make a random population, with contacts.

    This function returns a "popdict" dictionary, which has the following (required) keys:

        - uid: an array of (usually consecutive) integers of length N, uniquely identifying each agent
        - age: an array of floats of length N, the age in years of each agent
        - sex: an array of integers of length N (not currently used, so does not have to be binary)
        - contacts: list of length N listing the contacts; see make_random_contacts() for details
        - layer_keys: a list of strings representing the different contact layers in the population; see make_random_contacts() for details

    Args:
        pars (dict): the parameter dictionary or simulation object
        use_age_data (bool): whether to use location-specific age data
        use_household_data (bool): whether to use location-specific household size data
        sex_ratio (float): proportion of the population that is male (not currently used)
        microstructure (bool): whether or not to use the microstructuring algorithm to group contacts
        kwargs (dict): passed to contact creation method (e.g., make_hybrid_contacts)

    Returns:
        popdict (dict): a dictionary representing the population, with the following keys for a population of N agents with M contacts between them:
    '''

    pop_size = int(pars['pop_size']) # Number of people

    # Load age data and household demographics based on 2018 Seattle demographics by default, or country if available
    age_data = cvd.default_age_data
    location = pars['location']
    if location is not None:
        if pars['verbose']:
            print(f'Loading location-specific data for "{location}"')
        if use_age_data:
            try:
                age_data = cvdata.get_age_distribution(location)
            except ValueError as E:
                warnmsg = f'Could not load age data for requested location "{location}" ({str(E)}), using default'
                cvm.warn(warnmsg)
        if use_household_data:
            try:
                household_size = cvdata.get_household_size(location)
                if 'h' in pars['contacts']:
                    pars['contacts']['h'] = household_size - 1 # Subtract 1 because e.g. each person in a 3-person household has 2 contacts
                elif pars['verbose']:
                    keystr = ', '.join(list(pars['contacts'].keys()))
                    warnmsg = f'Not loading household size for "{location}" since no "h" key; keys are "{keystr}". Try "hybrid" population type?'
                    cvm.warn(warnmsg)
            except ValueError as E:
                if pars['verbose']>1: # These don't exist for many locations, so skip the warning by default
                    warnmsg = f'Could not load household size data for requested location "{location}" ({str(E)}), using default'
                    cvm.warn(warnmsg)

    # Handle sexes and ages
    uids           = np.arange(pop_size, dtype=cvd.default_int)
    sexes          = np.random.binomial(1, sex_ratio, pop_size)
    age_data_min   = age_data[:,0]
    age_data_max   = age_data[:,1] + 1 # Since actually e.g. 69.999
    age_data_range = age_data_max - age_data_min
    age_data_prob  = age_data[:,2]
    age_data_prob /= age_data_prob.sum() # Ensure it sums to 1
    age_bins       = cvu.n_multinomial(age_data_prob, pop_size) # Choose age bins
    ages           = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(pop_size) # Uniformly distribute within this age bin
    population_data = load_population_regions(pars['boundary_shapefile'])
    regions = assign_regions(pars, population_data)
    coords = assign_coords(regions, population_data)
    pars['people_coords'] = coords

    # Store output
    popdict = {}
    popdict['uid'] = uids
    popdict['age'] = ages
    popdict['sex'] = sexes
    popdict['region'] = regions

    # Actually create the contacts
    if microstructure == 'random':
        contacts = dict()
        for lkey, n in pars['contacts'].items():
            contacts[lkey] = make_random_contacts(pop_size, n, **kwargs)
    elif microstructure == 'hybrid':
        contacts = make_hybrid_contacts(pop_size, ages, pars['contacts'], **kwargs)
    elif microstructure == 'rank':
        coords = assign_density_based_coords(pars)                       # (N,2) x,y in km, sampled from the density raster
        popdict['x'], popdict['y'] = coords[:, 0], coords[:, 1]
        contacts = dict()
        for lkey,n in pars['contacts'].items():
            contacts[lkey] = make_rank_based_contacts(coords, n, **kwargs)
    else: # pragma: no cover
        errormsg = f'Microstructure type "{microstructure}" not found; choices are random or hybrid or rank'
        raise NotImplementedError(errormsg)

    popdict['contacts']   = contacts
    popdict['layer_keys'] = list(pars['contacts'].keys())

    return popdict


def _tidy_edgelist(p1, p2, mapping):
    ''' Helper function to convert lists to arrays and optionally map arrays '''
    p1 = np.array(p1, dtype=cvd.default_int)
    p2 = np.array(p2, dtype=cvd.default_int)
    if mapping is not None:
        mapping = np.array(mapping, dtype=cvd.default_int)
        p1 = mapping[p1]
        p2 = mapping[p2]
    output = dict(p1=p1, p2=p2)
    return output


def make_random_contacts(pop_size, n, overshoot=1.2, dispersion=None, mapping=None):
    '''
    Make random static contacts for a single layer as an edgelist.

    Args:
        pop_size   (int)   : number of agents to create contacts between (N)
        n          (int) : the average number of contacts per person for this layer
        overshoot  (float) : to avoid needing to take multiple Poisson draws
        dispersion (float) : if not None, use a negative binomial distribution with this dispersion parameter instead of Poisson to make the contacts
        mapping    (array) : optionally map the generated indices onto new indices

    Returns:
        Dictionary of two arrays defining UIDs of the edgelist (sources and targets)

    New in 3.1.1: optimized and updated arguments.
    '''

    # Preprocessing
    pop_size = int(pop_size) # Number of people
    p1 = [] # Initialize the "sources"
    p2 = [] # Initialize the "targets"

    # Precalculate contacts
    n_all_contacts  = int(pop_size*n*overshoot) # The overshoot is used so we won't run out of contacts if the Poisson draws happen to be higher than the expected value
    all_contacts    = cvu.choose_r(max_n=pop_size, n=n_all_contacts) # Choose people at random
    if dispersion is None:
        p_count = cvu.n_poisson(n, pop_size) # Draw the number of Poisson contacts for this person
    else:
        p_count = cvu.n_neg_binomial(rate=n, dispersion=dispersion, n=pop_size) # Or, from a negative binomial
    p_count = np.array((p_count/2.0).round(), dtype=cvd.default_int)

    # Make contacts
    count = 0
    for p in range(pop_size):
        n_contacts = p_count[p]
        these_contacts = all_contacts[count:count+n_contacts] # Assign people
        count += n_contacts
        p1.extend([p]*n_contacts)
        p2.extend(these_contacts)

    # Tidy up
    output = _tidy_edgelist(p1, p2, mapping)

    return output


def make_microstructured_contacts(pop_size, cluster_size, mapping=None):
    '''
    Create microstructured contacts -- i.e. for households.

    Args:
        pop_size (int): total number of people
        cluster_size (int): the average size of each cluster (Poisson-sampled)

    New in version 3.1.1: optimized updated arguments.
    '''

    # Preprocessing -- same as above
    pop_size = int(pop_size) # Number of people
    p1 = [] # Initialize the "sources"
    p2 = [] # Initialize the "targets"

    # Initialize
    n_remaining = pop_size # Make clusters - each person belongs to one cluster

    # Loop over the clusters
    cluster_id = -1
    while n_remaining > 0:
        cluster_id += 1 # Assign cluster id
        this_cluster =  cvu.poisson(cluster_size)  # Sample the cluster size
        if this_cluster > n_remaining:
            this_cluster = n_remaining

        # Indices of people in this cluster
        cluster_indices = (pop_size-n_remaining) + np.arange(this_cluster)
        for source in cluster_indices: # Add symmetric pairwise contacts in each cluster
            targets = set()
            for target in cluster_indices:
                if target > source:
                    targets.add(target)
            p1.extend([source]*len(targets))
            p2.extend(list(targets))

        n_remaining -= this_cluster

    # Tidy up
    output = _tidy_edgelist(p1, p2, mapping)

    return output


def make_hybrid_contacts(pop_size, ages, contacts, school_ages=None, work_ages=None):
    '''
    Create "hybrid" contacts -- microstructured contacts for households and
    random contacts for schools and workplaces, both of which have extremely
    basic age structure. A combination of both make_random_contacts() and
    make_microstructured_contacts().
    '''

    # Handle inputs and defaults
    contacts = sc.mergedicts({'h':4, 's':20, 'w':20, 'c':20}, contacts) # Ensure essential keys are populated
    if school_ages is None:
        school_ages = [6, 22]
    if work_ages is None:
        work_ages   = [22, 65]

    contacts_dict = {}

    # Start with the household contacts for each person
    contacts_dict['h'] = make_microstructured_contacts(pop_size, contacts['h'])

    # Make community contacts
    contacts_dict['c'] = make_random_contacts(pop_size, contacts['c'])

    # Get the indices of people in each age bin
    ages = np.array(ages)
    s_inds = sc.findinds((ages >= school_ages[0]) * (ages < school_ages[1]))
    w_inds = sc.findinds((ages >= work_ages[0])   * (ages < work_ages[1]))

    # Create the school and work contacts for each person
    contacts_dict['s'] = make_random_contacts(len(s_inds), contacts['s'], mapping=s_inds)
    contacts_dict['w'] = make_random_contacts(len(w_inds), contacts['w'], mapping=w_inds)

    return contacts_dict


def make_synthpop(sim=None, popdict=None, layer_mapping=None, community_contacts=None, **kwargs): # pragma: no cover
    '''
    Make a population using SynthPops, including contacts.

    Usually called automatically, but can also be called manually. Either a simulation
    object or a population must be supplied; if a population is supplied, transform
    it into the correct format; otherwise, create the population and then transform it.

    Args:
        sim (Sim): a Covasim simulation object
        popdict (dict/Pop/People): a pre-generated SynthPops population (otherwise, create a new one)
        layer_mapping (dict): a custom mapping from SynthPops layers to Covasim layers
        community_contacts (int): if a simulation is not supplied, create this many community contacts on average
        kwargs (dict): passed to sp.make_population()

    **Example**::

        sim = cv.Sim(pop_type='synthpops')
        sim.popdict = cv.make_synthpop(sim)
        sim.run()
    '''
    try:
        import synthpops as sp # Optional import
    except ModuleNotFoundError as E: # pragma: no cover
        errormsg = 'Please install the optional SynthPops module first, e.g. pip install synthpops' # Also caught in make_people()
        raise ModuleNotFoundError(errormsg) from E

    # Handle layer mapping
    default_layer_mapping = {'H':'h', 'S':'s', 'W':'w', 'C':'c', 'LTCF':'l'} # Remap keys from old names to new names
    layer_mapping = sc.mergedicts(default_layer_mapping, layer_mapping)

    # Handle community contacts
    if community_contacts is None:
        if sim is not None:
            community_contacts = sim['contacts']['c']
        else: # pragma: no cover
            errormsg = 'If a simulation is not supplied, the number of community contacts must be specified'
            raise ValueError(errormsg)

    # Main use case -- generate a new population
    pop_size = sim['pop_size']
    if popdict is None:
        if sim is None: # pragma: no cover
            errormsg = 'Either a simulation or a population must be supplied'
            raise ValueError(errormsg)
        people = sp.Pop(n=pop_size, rand_seed=sim['rand_seed'], **kwargs).to_people() # Actually generate it
    else: # Otherwise, convert to a sp.People object (similar to a cv.People object)
        if isinstance(popdict, sp.people.People):
            people = popdict
        elif isinstance(popdict, sp.Pop):
            people = popdict.to_people()
        elif isinstance(popdict, dict):
            people = sp.people.make_people(popdict=popdict)
        elif isinstance(popdict, cvb.BasePeople):
            return popdict # Already the right format
        else:
            errormsg = f'Cannot understand population of type {type(popdict)}: must be dict, sp.Pop, sp.People, or cv.People'
            raise TypeError(errormsg)

    # Convert contacts from SynthPops to Covasim
    people.contacts = cvb.Contacts(**people.contacts)

    # Add community contacts and layer keys
    c_contacts = make_random_contacts(pop_size, community_contacts)
    people.contacts.add_layer(c=c_contacts)
    people['layer_keys'] = list(layer_mapping.values())

    return people

# ----------------------------------------------------------------------------
# 2. The rank-based network
# ----------------------------------------------------------------------------

def make_rank_based_contacts_basic(coords, n, alpha=1.0, dispersion=None, mapping=None):

    '''
    Make a rank-based geographic friendship layer as an edgelist.

    Model: Liben-Nowell et al. (2005), "Geographic routing in social networks".
    Person u links to person v with probability proportional to

        P(u -> v)  ~  1 / rank_u(v)^alpha,    alpha = 1 in the paper

    where rank_u(v) is the number of people w (w != u) who are at least as close to u
    as v is. Rank (not distance) means links are short in dense areas and long in
    sparse areas.

    Args:
        coords     (array) : (N,2) array of x,y locations in km, one row per person
        n          (float) : the average number of friends per person for this layer
        alpha      (float) : how strongly friendships favour nearby people (0 = ignore
                             distance, 1.0 = the value from Liben-Nowell et al., larger = more local)
        dispersion (float) : if not None, use a negative binomial distribution with this
                             dispersion parameter instead of Poisson for friend counts
        mapping    (array) : optionally map the generated indices onto new indices

    Returns:
        Dictionary of two arrays, p1 and p2, defining the edgelist (friend pairs), in the
        same format as make_random_contacts() so it can be used as a contact layer.

    Method, for each person p:
        1. Draw how many friendships p starts (Poisson, averaging about n/2, since each
           friendship has two ends).
        2. For each one, draw a rank r in 1..N-1 with probability proportional to 1/r^alpha.
        3. Link p to their r-th nearest person (found by partially sorting the distances to everyone).
    Duplicate and mutual pairs (A-B and B-A) are then merged, so the mean degree comes out
    slightly below n (about 0.9n to 0.95n).

    Method, for each person p:
        1. Draw how many friends p picks: a random (Poisson) number averaging n/2. Every
           friendship is picked by one of its two people, so a person's total comes from the
           n/2 they pick plus about n/2 picked by others, about n friends in all.
        2. For each friendship, pick a rank r from 1 to N-1. Small ranks are much more
           likely than large ones (probability proportional to 1/r^alpha).
        3. Connect p to the person who is r-th nearest to them: sort everyone else by
           distance from p and take the r-th one. (The code only partially sorts, which is faster.)
        4. Finally, pairs found twice (A-B and B-A, or A-B repeated) are merged into one friendship.
           This is why the average number of friends comes out slightly below n (about 0.9n to 0.95n).

    Notes:
         - Cost grows roughly with N^2. For each of the N people, the code computes the
          distance to everyone else (about N calculations) and then partially sorts that list
          (about N more), so the total is about N x N. This takes about 6 s for 10,000
          people and 10 minutes for 100,000. make_random_contacts() never looks at distances,
          so its time grows roughly in step with N.
    
    '''

    # Preprocessing
    coords   = np.asarray(coords, dtype=float)
    pop_size = len(coords) # Number of people
    p1 = [] # Initialize the "sources"
    p2 = [] # Initialize the "targets"
    if pop_size < 2: # Nobody to befriend
        return dict(p1=np.array([], dtype=cvd.default_int), p2=np.array([], dtype=cvd.default_int))

    # Precalculate the probability of each rank
    # Rank 1 = nearest person, rank pop_size-1 = farthest. Weight of rank r is 1/r^alpha.
    rank_weights = np.arange(1, pop_size, dtype=float)**(-alpha)
    rank_cdf     = np.cumsum(rank_weights) # Running total, used to turn random numbers into ranks
    rank_cdf    /= rank_cdf[-1]

    # Precalculate how many friendships each person starts (same approach as make_random_contacts)
    if dispersion is None:
        p_count = cvu.n_poisson(n, pop_size) # Draw the number of Poisson contacts for this person
    else:
        p_count = cvu.n_neg_binomial(rate=n, dispersion=dispersion, n=pop_size) # Or, from a negative binomial
    p_count = np.array((p_count/2.0).round(), dtype=cvd.default_int) # Halve: each friendship has two ends

    # Make contacts
    for p in range(pop_size):
        n_friends = p_count[p]
        if n_friends == 0:
            continue

        # Pick a rank for each friendship (1 = nearest person); rank_cdf maps random numbers to ranks
        ranks = np.searchsorted(rank_cdf, np.random.random(n_friends)) + 1
        ranks = np.minimum(ranks, pop_size-1) # Guard against rounding at the far end

        # Squared distance from this person to everyone (squared is fine for ordering)
        dist2 = ((coords - coords[p])**2).sum(axis=1)
        dist2[p] = np.inf # Put self last so it can never be chosen

        # Find who sits at each chosen rank; argpartition avoids sorting everyone
        order   = np.argpartition(dist2, np.unique(ranks-1))
        friends = order[ranks-1]

        p1.append(np.full(n_friends, p))
        p2.append(friends)

    # Tidy up: join the pieces, then merge duplicate pairs (A-B and B-A are the same friendship)
    p1 = np.concatenate(p1) if p1 else np.empty(0)
    p2 = np.concatenate(p2) if p2 else np.empty(0)
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    pairs  = np.unique(np.column_stack((lo, hi)), axis=0)
    p1, p2 = pairs[:, 0], pairs[:, 1]

    if mapping is not None:
        mapping = np.asarray(mapping)
        p1, p2  = mapping[p1], mapping[p2]

    output = dict(p1=p1.astype(cvd.default_int), p2=p2.astype(cvd.default_int))
    return output

#*******************************************
# RANK MODEL - FAST
#***********************************************


#Fast version of the rank-based friendship network (same result as rank_network.py, about 3 times faster on one core,
#and it can use several cores: pass use_workers=True and it chooses how many for this computer (or n_workers=N to choose yourself).

#The main function is called make_rank_based_contacts_fast() here, so it cannot be confused with the original
#make_rank_based_contacts() while both are being tested. In covasim/population.py it takes the name make_rank_based_contacts().

#This file is for testing next to rank_network.py. In Covasim itself, the same two helper functions and
#
# make_rank_based_contacts() live in covasim/population.py.

_SHARED = {} # The coordinates, shared with worker processes that are started by fork (so they are not sent again with every job)
_MIN_AGENTS_PER_WORKER = 2000 # A rough guide: fewer agents than this per worker and starting the workers costs more than it saves
_AUTO_MAX_WORKERS = 16 # use_workers=True never picks more than this: on the test server (100,000 agents) efficiency stayed above about 80% up to 16 workers


def choose_workers(pop_size, use_workers=False, n_workers=None):
    '''
    Helper for make_rank_based_contacts(): how many worker processes to use. One means no parallelism.
        n_workers given    use that many (-1 = every core this job may use), but never more than the cores available or the
                           size justifies (about 2,000 agents per worker)
        use_workers=True   choose automatically: leave one core free, never more than 16 (more cores help less and less),
                           and only count cores that are not already busy with other work
        neither            one core
    '''
    import os
    usable = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1) # Cores this job may really use
    if n_workers is not None:
        wanted = usable if n_workers < 0 else int(n_workers)
    elif use_workers:
        wanted = usable - 1 if usable > 2 else usable
        wanted = min(wanted, _AUTO_MAX_WORKERS)
        try:
            wanted = min(wanted, int(usable - os.getloadavg()[0])) # Cores already busy with other work are not free
        except (AttributeError, OSError): # getloadavg does not exist on every platform
            pass
    else:
        wanted = 1
    return max(1, min(wanted, usable, pop_size//_MIN_AGENTS_PER_WORKER))


def pick_ranks(d, ranks):
    '''
    Helper for make_rank_based_contacts(): given the squared distances d from one person to everyone, return who sits
    at each wanted rank (1 = nearest). The largest rank is found first with one partial sort; the smaller ranks are then
    searched only among the people already known to be closer, so each later step works on a smaller list.
    '''
    uniq, inverse = np.unique(ranks, return_inverse=True) # Wanted ranks, smallest first (a rank asked for twice is found once)
    idx   = np.arange(len(d)) # Who each entry of d belongs to (d gets reordered below)
    found = np.empty(len(uniq), dtype=np.int64)
    for j in range(len(uniq)-1, -1, -1): # Largest rank first
        r     = uniq[j]
        part  = np.argpartition(d, r-1) # Puts the r-th nearest at position r-1, the r nearest before it
        idx, d = idx[part], d[part]
        found[j] = idx[r-1]
        idx, d = idx[:r], d[:r] # The smaller ranks can only be among these r people
    return found[inverse]


def find_friends(x, y, people, counts, ranks):
    '''
    Helper for make_rank_based_contacts(): for each person in `people`, find the person at each of their wanted ranks.
    ranks holds the wanted ranks of all these people one after another (counts[i] of them for people[i]).
    '''
    friends = np.empty(len(ranks), dtype=np.int64)
    dx = np.empty(len(x)) # Reused for every person, so no new arrays are made inside the loop
    dy = np.empty(len(x))
    start = 0
    for p, k in zip(people, counts):
        np.subtract(x, x[p], out=dx)
        np.subtract(y, y[p], out=dy)
        np.multiply(dx, dx, out=dx)
        np.multiply(dy, dy, out=dy)
        np.add(dx, dy, out=dx) # Squared distance from person p to everyone (squared is fine for ordering)
        dx[p] = np.inf # Put self last so it can never be chosen
        friends[start:start+k] = pick_ranks(dx, ranks[start:start+k])
        start += k
    return friends


def friends_job(people, counts, ranks):
    '''
    Helper for make_rank_based_contacts(): one job for a worker process. The coordinates come from _SHARED, which the
    worker inherited when it was forked, so only the small per-job arrays have to be sent to it.
    '''
    return find_friends(_SHARED['x'], _SHARED['y'], people, counts, ranks)


def make_rank_based_contacts(coords, n, alpha=1.0, dispersion=None, mapping=None, use_workers=False, n_workers=None):
    '''
    Make a rank-based geographic friendship layer as an edgelist.

    Model: Liben-Nowell et al. (2005), "Geographic routing in social networks".
    Person u links to person v with probability proportional to

        P(u -> v)  ~  1 / rank_u(v)^alpha,    alpha = 1 in the paper

    where rank_u(v) is the number of people w (w != u) who are at least as close to u
    as v is. Rank (not distance) means links are short in dense areas and long in
    sparse areas.

    Args:
        coords     (array) : (N,2) array of x,y locations in km, one row per person
        n          (float) : the average number of friends per person for this layer
        alpha      (float) : how strongly friendships favour nearby people (0 = ignore
                             distance, 1.0 = the value from Liben-Nowell et al., larger = more local)
        dispersion (float) : if not None, use a negative binomial distribution with this
                             dispersion parameter instead of Poisson for friend counts
        mapping    (array) : optionally map the generated indices onto new indices
        use_workers (bool) : False (the default) uses one core, which works everywhere. True lets the code choose how many
                             cores to use for this computer and this size (see below).
        n_workers  (int)   : use this many cores instead (-1 = all the cores this job may use). Overrides use_workers.
                             Never more than the cores available or about one per 2,000 agents.
                             The result is identical whatever the number of cores.

    Returns:
        Dictionary of two arrays, p1 and p2, defining the edgelist (friend pairs), in the
        same format as make_random_contacts() so it can be used as a contact layer.

    Method, for each person p:
        1. Draw how many friends p picks: a random (Poisson) number averaging n/2. Every
           friendship is picked by one of its two people, so a person's total comes from the
           n/2 they pick plus about n/2 picked by others, about n friends in all.
        2. For each friendship, pick a rank r from 1 to N-1. Small ranks are much more
           likely than large ones (probability proportional to 1/r^alpha).
        3. Connect p to the person who is r-th nearest to them (found by partially sorting the
           distances from p to everyone, not a full sort).
        4. Finally, pairs found twice (A-B and B-A, or A-B repeated) are merged into one friendship.
           This is why the average number of friends comes out slightly below n (about 0.9n to 0.95n).
    
    Speed Improvements:
    - Arrays reused: the old code created temporary arrays for every person, whereas the new code allocates two arrays once and reuses them.
    - Separate x and y arrays: the coordinates are split into one-dimensional arrays rather than one 2D array, making the element-wise calculations more efficient.
    - Duplicates removed as integers: each friendship is temporarily encoded as a single integer while duplicates are removed (then decoded back, so the output is unchanged), which is faster than comparing rows in a 2D array as in the original.

    Notes:
        - use_workers=True picks the number of workers itself: one core left free, at most 16, only cores that are not
          already busy, and about one worker per 2,000 agents. Below about 4,000 agents it stays on one core. If worker
          processes cannot be started (for example inside MultiSim) it warns and uses one core.
        - All random numbers are drawn first, in one step, so the result does not depend on the number of workers.
        - 
    '''

    # Preprocessing
    coords   = np.asarray(coords, dtype=float)
    pop_size = len(coords) # Number of people
    empty    = dict(p1=np.array([], dtype=cvd.default_int), p2=np.array([], dtype=cvd.default_int))
    if pop_size < 2: # Nobody to befriend
        return empty

    # Precalculate the probability of each rank
    # Rank 1 = nearest person, rank pop_size-1 = farthest. Weight of rank r is 1/r^alpha.
    rank_weights = np.arange(1, pop_size, dtype=float)**(-alpha)
    rank_cdf     = np.cumsum(rank_weights) # Running total, used to turn random numbers into ranks
    rank_cdf    /= rank_cdf[-1]

    # Precalculate how many friendships each person starts (same approach as make_random_contacts)
    if dispersion is None:
        p_count = cvu.n_poisson(n, pop_size) # Draw the number of Poisson contacts for this person
    else:
        p_count = cvu.n_neg_binomial(rate=n, dispersion=dispersion, n=pop_size) # Or, from a negative binomial
    p_count = np.array((p_count/2.0).round(), dtype=cvd.default_int) # Halve: each friendship has two ends

    # Draw every rank up front (the same random numbers, in the same order, as drawing them person by person)
    people = np.flatnonzero(p_count > 0) # Who starts any friendships
    counts = p_count[people].astype(np.int64)
    n_draws = int(counts.sum())
    if n_draws == 0:
        return empty
    ranks = np.searchsorted(rank_cdf, np.random.random(n_draws)) + 1 # Rank 1 = nearest person
    ranks = np.minimum(ranks, pop_size-1) # Guard against rounding at the far end

    # Find who sits at each chosen rank, on one core or several
    x, y = np.ascontiguousarray(coords[:, 0]), np.ascontiguousarray(coords[:, 1])
    n_workers = choose_workers(pop_size, use_workers, n_workers) # One core unless asked otherwise

    friends = None
    if n_workers > 1:
        import multiprocessing as mp
        if 'fork' in mp.get_all_start_methods(): # Forked workers inherit the coordinates; other platforms use one core
            from concurrent.futures import ProcessPoolExecutor
            ctx    = mp.get_context('fork')
            ends   = np.cumsum(counts)
            starts = ends - counts
            groups = np.array_split(np.arange(len(people)), n_workers*4) # Several small jobs per core keeps all cores busy
            jobs   = [g for g in groups if len(g)]
            _SHARED['x'], _SHARED['y'] = x, y
            try:
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                    futures = [pool.submit(friends_job, people[g], counts[g], ranks[starts[g[0]]:ends[g[-1]]]) for g in jobs]
                    friends = np.concatenate([f.result() for f in futures])
            except (AssertionError, OSError, RuntimeError) as E: # e.g. already inside a worker process (MultiSim), or not allowed to start processes
                import warnings
                warnings.warn(f'Could not start worker processes for the rank network ({E}); using one core instead')
                friends = None
            finally:
                _SHARED.clear()
    if friends is None: # One core: the default, and the fallback
        friends = find_friends(x, y, people, counts, ranks)

    # Tidy up: merge duplicate pairs (A-B and B-A are the same friendship)
    p1 = np.repeat(people, counts)
    p2 = friends
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    pairs  = np.unique(lo*pop_size + hi) # One number per pair, so unique() is much faster than comparing rows
    p1, p2 = pairs//pop_size, pairs % pop_size

    if mapping is not None:
        mapping = np.asarray(mapping)
        p1, p2  = mapping[p1], mapping[p2]

    output = dict(p1=p1.astype(cvd.default_int), p2=p2.astype(cvd.default_int))
    return output

    