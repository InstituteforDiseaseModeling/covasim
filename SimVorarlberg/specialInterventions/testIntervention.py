import sciris as sc
from covasim import utils as cvu
from covasim import base as cvb
from covasim import Intervention
from  covasim.interventions import process_changes, process_days, process_daily_data, find_day

class change_beta_by_age(Intervention):
    '''
    Isolate contacts by removing them from the simulation. Contacts are treated as
    "edges", and this intervention works by removing them from sim.people.contacts
    and storing them internally. When the intervention is over, they are moved back.
    Similar to change_beta().

    Args:
        days (int or array): the day or array of days to isolate contacts
        changes (float or array): the changes in the number of contacts (1 = no change, 0 = no contacts)
        layers (str or list): the layers in which to isolate contacts (if None, then all layers)
        kwargs (dict): passed to Intervention()

    **Examples**::

        interv = cv.clip_edges(25, 0.3) # On day 25, reduce overall contacts by 70% to 0.3
        interv = cv.clip_edges([14, 28], [0.7, 1], layers='w') # On day 14, remove 30% of school contacts, and on day 28, restore them
    '''

    def __init__(self, days, changes, age=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        self.days     = sc.dcp(days)
        self.changes  = sc.dcp(changes)
        self.age   = sc.dcp(age)
        self.contacts = None
        return


    def initialize(self, sim):
        self.days    = process_days(sim, self.days)
        self.changes = process_changes(sim, self.changes, self.days)
        if self.age is None:
            self.age = 50

        self.contacts = cvb.Contacts(layer_keys=None) #all Layers
        self.initialized = True
        return


    def apply(self, sim):

        # If this day is found in the list, apply the intervention
        for ind in find_day(self.days, sim.t):

            for lkey in sim.people.contacts.keys():

                s_layer= sim.people.contacts[lkey]

                i = 0

                for person in s_layer['p1']:
                    age_of_p1 = sim.people.age[person]

                    if age_of_p1 >= self.age:
                        s_layer['beta'][i] = self.changes

                    i = i+1

                i = 0

                for person in s_layer['p2']:
                    age_of_p2 = sim.people.age[person]

                    if age_of_p2 >= self.age:
                        s_layer['beta'][i] = self.changes

                    i = i + 1

        # Ensure the edges get deleted at the end
        if sim.t == sim.tvec[-1]:
            self.contacts = None # Reset to save memory

        return