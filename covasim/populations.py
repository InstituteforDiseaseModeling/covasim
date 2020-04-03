import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import utils as cvu
from . import parameters as cvpars
from . import requirements as cvreqs
from . import utils as cvu
from .people import Person

class ContactLayer(sc.prettyobj):
    """

    Beta is stored as a single scalar value so that it can be overwritten or otherwise
    modified by interventions in a consistent fashion

    """

    def __init__(self, name:str, beta: float, traceable: bool=True) -> None:
        self.name = name  #: Name of the contact layer e.g. 'Households'
        self.beta = beta  #: Transmission probability per contact (absolute)
        self.traceable = traceable  #: If True, the contacts should be considered tracable via contact tracing
        return

    def get_contacts(self, person, sim) -> list:
        """
        Get contacts for a person

        :param person:
        :param sim: The simulation instance
        :return: List of contact *indexes* e.g. [1,50,295]

        """
        raise NotImplementedError


class StaticContactLayer(ContactLayer):
    def __init__(self, name:str, beta: float, contacts: dict) -> None:
        """
        Contacts that are the same every timestep

        Suitable for groups of people that do not change over time e.g., households, workplaces

        Args:
            name:
            beta:
            contacts:

        """

        super().__init__(name, beta)
        self.contacts = contacts  #: Dictionary mapping `{source UID:[target indexes]}` storing interactions
        return

    def get_contacts(self, person, sim) -> list:
        return self.contacts[person.uid]


class RandomContactLayer(ContactLayer):
    def __init__(self, name:str, beta: float, max_n: int, n: int) -> None:
        """
        Randomly sampled contacts each timestep

        Suitable for interactions that randomly occur e.g., community transmission

        Args:
            name:
            beta: Transmission probability per contact (absolute)
            max_n: Number of people available
            n: Number of contacts per person

        """

        super().__init__(name, beta, traceable=False) # nb. cannot trace random contacts e.g. in community
        self.max_n = max_n
        self.n = n  #: Number of randomly sampled contacts per timestep

    def get_contacts(self, person, sim) -> list:
        return cvu.choose(max_n=self.max_n, n=self.n)


def make_random_population(pars, n_people: int = 2000, n_regular_contacts: int = 20, n_random_contacts: int = 0, id_len=6):
    """
    Make a simple random population

    Args:
        pars: Simulation parameters
        n_people: Number of people in population
        n_infected: Number of seed infections
        n_regular_contacts: Regular/repeat number of contacts (e.g. household size)
        n_random_contacts: Number of random contacts (e.g. community encounters per day)
        id_len: Optionally specify UUID length (may be necessary if requesting a very large number of people)

    Returns: Tuple containing (people: dict, contact layers: dict)

    """

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
        [75, 79, 0.02016],  # Calculated based on 0.0504 total for >=75
        [80, 84, 0.01344],
        [85, 89, 0.01008],
        [90, 99, 0.00672],
        ])

    # Handle sex and UID
    uids = sc.uuid(which='ascii', n=n_people, length=id_len, tostring=True)
    sexes = cvu.rbt(0.5, n_people)

    # Handle ages
    age_data_min  = age_data[:,0]
    age_data_max  = age_data[:,1] + 1 # Since actually e.g. 69.999
    age_data_range = age_data_max - age_data_min
    age_data_prob = age_data[:,2]
    age_data_prob /= age_data_prob.sum() # Ensure it sums to 1
    age_bins = cvu.mt(age_data_prob, n_people) # Choose age bins
    ages = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(n_people) # Uniformly distribute within this age bin

    # Instantiate people
    people = {uid:Person(pars=pars, uid=uid, age=age, sex=sex) for uid, age, sex in zip(uids, ages, sexes)}

    # Make contacts
    contact_layers = {}

    # Make static contact matrix
    contacts = {}
    for i, person in enumerate(people.values()):
        n_contacts = cvu.pt(n_regular_contacts)  # Draw the number of Poisson contacts for this person
        contacts[person.uid] = cvu.choose(max_n=n_people, n=n_contacts)  # Choose people at random, assigning to 'household'
    layer = StaticContactLayer(name='Household', beta=0.015, contacts=contacts)
    contact_layers[layer.name] = layer

    # Make random contacts
    contact_layers['Community'] = RandomContactLayer(name='Community', beta=0.015, max_n=n_people, n=n_random_contacts)

    return people, contact_layers


# def make_people(sim, verbose=None, id_len=None, die=True, reset=False):
#     '''
#     Make the actual people for the simulation.
#
#     Args:
#         sim (Sim): the simulation object
#         verbose (bool): level of detail to print
#         id_len (int): length of ID for each person (default: calculate required length based on the number of people)
#         die (bool): whether or not to fail if synthetic populations are requested but not available
#         reset (bool): whether to force population creation even if self.popdict exists
#
#     Returns:
#         None.
#     '''
#
#     # Set inputs
#     n_people     = int(sim['n']) # Shorten
#     usepopdata   = sim['usepopdata'] # Shorten
#     use_rand_pop = (usepopdata == 'random') # Whether or not to use a random population (as opposed to synthpops)
#
#     # Set defaults
#     if verbose is None: verbose = sim['verbose']
#     if id_len  is None: id_len  = int(np.log10(n_people)) + 2 # Dynamically generate based on the number of people required
#
#     # Check which type of population to rpoduce
#     if not use_rand_pop and not cvreqs.available['synthpops']:
#         errormsg = f'You have requested "{usepopdata}" population, but synthpops is not available; please use "random"'
#         if die:
#             raise ValueError(errormsg)
#         else:
#             print(errormsg)
#             usepopdata = 'random'
#
#     # Actually create the population
#     if sim.popdict and not reset:
#         popdict = sim.popdict # Use stored one
#     else:
#         if use_rand_pop:
#             popdict = make_randpop(sim)
#         else:
#             import synthpops as sp # Optional import
#             population = sp.make_population(n=sim['n'])
#             uids, ages, sexes, contacts = [], [], [], []
#             for uid,person in population.items():
#                 uids.append(uid)
#                 ages.append(person['age'])
#                 sexes.append(person['sex'])
#
#             # Replace contact UIDs with ints...
#             for uid,person in population.items():
#                 uid_contacts = person['contacts']
#                 int_contacts = {}
#                 for key in uid_contacts.keys():
#                     int_contacts[key] = []
#                     for uid in uid_contacts[key]:
#                         int_contacts[key].append(uids.index(uid))
#                     int_contacts[key] = np.array(int_contacts[key], dtype=np.int64)
#                 contacts.append(int_contacts)
#
#             popdict = {}
#             popdict['uid']      = uids
#             popdict['age']      = np.array(ages)
#             popdict['sex']      = np.array(sexes)
#             popdict['contacts'] = contacts
#
#
#     # Actually create the people
#     people = {} # Dictionary for storing the people -- use plain dict since faster than odict
#     for p in range(n_people): # Loop over each person
#         keys = ['uid', 'age', 'sex', 'contacts', 'symp_prob', 'severe_prob', 'crit_prob', 'death_prob']
#         person_args = {}
#         for key in keys:
#             person_args[key] = popdict[key][p] # Convert from list to dict
#         person = Person(pars=sim.pars, **person_args) # Create the person
#         people[person_args['uid']] = person # Save them to the dictionary
#
#     # Store UIDs and people
#     sim.popdict = popdict
#     sim.uids = popdict['uid'] # Duplication, but used in an innermost loop so make as efficient as possible
#     sim.people = people
#     sim.contact_keys = list(sim['contacts_pop'].keys())
#
#     average_age = sum(popdict['age']/n_people)
#     sc.printv(f'Created {n_people} people, average age {average_age:0.2f} years', 1, verbose)
#
#     return
