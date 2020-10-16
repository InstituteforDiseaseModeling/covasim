import sciris as sc
from covasim import utils as cvu
from covasim import base as cvb
from covasim import Intervention
from  covasim.interventions import process_changes, process_days, process_daily_data, find_day

class change_beta_by_age(Intervention):
    '''
    Change beta (transmission) by a certain amount on a given day or days, constrained by the persons age

    Args:
        days (int or array): the day or array of days to isolate contacts
        changes (float or array): the changes in the number of contacts (1 = no change, 0 = no contacts)
        age (int): the age on which the beta should be changed
        kwargs (dict): passed to Intervention()

    **Example**::

        from SimVorarlberg.specialInterventions.testIntervention import change_beta_by_age

        change_beta_by_age(days=intervention_start_day, changes=0, age=55)
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