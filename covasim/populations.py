import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import utils as cvu
from . import parameters as cvpars
from . import requirements as cvreqs
from . import utils as cvu
from .people import Person

class Population(sc.prettyobj):
    """
    Class to represent a population of people

    A population is defined by
    - A collection of people (`Person` instances)
    - A collection of networks specifying how those people interact (a collection of `ContactLayer` instances)

    Thus this class essentially specifies the graph upon which infection and
    transmission take place

    """

    def __init__(self):
        self.people = {}  #: Store Person instances
        self.contact_layers = {}  #: Store ContactLayer instances
        self._uids = {}  #: Map index to UID

    def get_person(self, ind):
        ''' Return a person based on their ID '''
        return self.people[self._uids[ind]]

    @classmethod
    def random(cls, pars, n_people: int = 2000, n_regular_contacts: int = 20, n_random_contacts: int = 0, id_len=6):
        """
        Make a simple random population

        Args:
            pars: Simulation parameters
            n_people: Number of people in population
            n_infected: Number of seed infections
            n_regular_contacts: Regular/repeat number of contacts (e.g. household size)
            n_random_contacts: Number of random contacts (e.g. community encounters per day)
            id_len: Optionally specify UUID length (may be necessary if requesting a very large number of people)

        Returns: A Population instance

        """

        self = cls()

        # Load age data based on 2018 Seattle demographics
        age_data = np.array([
            [0, 4, 0.0605],
            [5, 9, 0.0607],
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
        age_data_min = age_data[:, 0]
        age_data_max = age_data[:, 1] + 1  # Since actually e.g. 69.999
        age_data_range = age_data_max - age_data_min
        age_data_prob = age_data[:, 2]
        age_data_prob /= age_data_prob.sum()  # Ensure it sums to 1
        age_bins = cvu.mt(age_data_prob, n_people)  # Choose age bins
        ages = age_data_min[age_bins] + age_data_range[age_bins] * np.random.random(n_people)  # Uniformly distribute within this age bin

        # Instantiate people
        self.people = {uid: Person(pars=pars, uid=uid, age=age, sex=sex) for uid, age, sex in zip(uids, ages, sexes)}
        self._uids = {i:x.uid for i,x in enumerate(self.people.values())}

        # Make contacts
        self.contact_layers = {}

        # Make static contact matrix
        contacts = {}
        for i, person in enumerate(self.people.values()):
            n_contacts = cvu.pt(n_regular_contacts)  # Draw the number of Poisson contacts for this person
            contacts[person.uid] = cvu.choose(max_n=n_people, n=n_contacts)  # Choose people at random, assigning to 'household'
        layer = StaticContactLayer(name='Household', beta=0.015, contacts=contacts)
        self.contact_layers[layer.name] = layer

        # Make random contacts
        self.contact_layers['Community'] = RandomContactLayer(name='Community', beta=0.015, max_n=n_people, n=n_random_contacts)

        return self

    @classmethod
    def synthpops(cls, pars, n_people = 5000, n_random_contacts: int = 20, beta = 0.015, betas = None):
        """
        Construct network with microstructure using Synthpops

        Args:
            pars: Covasim parameters (e.g. output from `covasim.make_pars()`) used when initializing people
            n_people: Number of people
            n_random_contacts: Number of random community contacts each day
            beta: Baseline beta value
            betas: Optionally specify dict with relative beta values for each contact layer

        Returns: A Population instance

        """

        if betas is None:
            betas = {'H': 1.7, 'S': 0.8, 'W': 0.8, 'R': 0.3}  # Per-population beta weights; relative

        import synthpops as sp  # Optional import
        population = sp.make_population(n_people)

        self = cls()

        # Make people
        self.people = {}
        for uid, person in population.items():
            self.people[uid] = Person(pars=pars, uid=uid, age=person['age'], sex=person['sex'])
        self._uids = {i:x.uid for i,x in enumerate(self.people.values())}

        # Make contact layers
        layers = ['H','S','W'] # Hardcode the expected synthpops contact layers for now
        self.contact_layers = {}
        uid_to_index = {x.uid:i for i,x in enumerate(self.people.values())}
        for layer in layers:
            contacts = {}
            for uid, person in population.items():
                contacts[uid] = np.array([uid_to_index[uid] for uid in person['contacts'][layer]],dtype=np.int64) # match datatype in covasim.utils.bf
                self.people[uid] = Person(pars=pars, uid=uid, age=person['age'], sex=person['sex'])
            self.contact_layers[layer] = StaticContactLayer(name=layer, beta=beta*betas[layer], contacts=contacts)
        self.contact_layers['R'] = RandomContactLayer(name='R', beta=beta*betas['R'], max_n=n_people, n=n_random_contacts)

        return self

    @classmethod
    def country(cls, country_code, beta = 0.015):
        """
        Create population from country data

        Args:
            country_code: ISO Country code to specify country e.g. 'IND', 'TZA'

        Returns: A Population instance

        """
        raise NotImplementedError

    @staticmethod
    def load(filename, **kwargs):
        '''
        Load the population dictionary from file.

        Args:
            filename (str): name of the file to load.
        '''
        filepath = sc.makefilepath(filename=filename, **kwargs)
        pop = sc.loadobj(filepath)
        if not isinstance(pop, Population):
            raise TypeError('Loaded file was not a population')
        return pop

    def save(self, filename, **kwargs):
        '''
        Save the population dictionary to file.

        Args:
            filename (str): name of the file to save to.
        '''
        filepath = sc.makefilepath(filename=filename, **kwargs)
        sc.saveobj(filepath, self)
        return filepath



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



