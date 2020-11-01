import sciris as sc
from covasim import base as cvb
from covasim import Intervention
from covasim.interventions import process_changes, process_days, find_day
import numpy as np


def get_contact_inds(ind, layer):
    inds = sc.promotetolist()

    for i in range(len(layer)):
        if layer['p1'][i] == ind or layer['p2'][i] == ind:
            inds.append(i)

    return inds


def process_prob(prob):
    if not isinstance(prob, float):
        if isinstance(prob, tuple([str, int])):
            prob = float(prob)
        else:
            raise TypeError(f'Probability must be of type float, or be convertible to it.')

    return prob


def process_layers(sim, layers):
    if not hasattr(layers, '__iter__'):
        layers = sc.promotetolist(layers)

    for lkey in layers:
        if not lkey in sim.people.contacts.keys():
            raise KeyError(f'{lkey} is not a valid key. Valid keys: {sim.people.contacts.keys()}')

    return layers


class selfequarantinOnSymptomatic_intervention(Intervention):

    def __init__(self, sq_prob_symp, layers=None, **kwargs):
        super().__init__(**kwargs)
        self.sq_prob_symp = sc.dcp(sq_prob_symp)
        self.layers = sc.dcp(layers)

    def initialize(self, sim):
        self.sq_prob_symp = process_prob(self.sq_prob_symp)
        if self.layers == None:
            self.layers = sc.dcp(sim.people.contacts.keys())
        else:
            self.layers = process_layers(sim, self.layers)
        self.symPeople = dict()
        self.initialized = True

    def apply(self, sim):

        t = sim.t

        new_symptomatic = sim.results['new_symptomatic'][max(t - 1, 0)]
        new_severe = sim.results['new_severe'][max(t - 1, 0)]
        new_critical = sim.results['new_critical'][max(t - 1, 0)]
        new_deaths = sim.results['new_deaths'][max(t - 1, 0)]
        new_recoveries = sim.results['new_recoveries'][max(t - 1, 0)]

        sum_ns_nc_nd_nr = new_recoveries + new_deaths + new_severe + new_critical

        counter1 = counter2 = 0

        #debug
        #print(new_symptomatic)
        #print(sum_ns_nc_nd_nr)

        if new_symptomatic > 0:

            for ind in sim.people.uid:

                if sim.people.symptomatic[ind]:

                    if ind not in self.symPeople.keys():
                        sim.people.diagnosed[ind] = True
                        counter1 = counter1 + 1

                        sim.people.quarantine(inds=[ind], start_date=t+1, period=14)

                        self.symPeople[ind] = symPerson(ind, sim, self.layers)
                        #contacts = dict()
#
                        #for lkey in sim.people.contacts.keys():
                        #    contacts[lkey] = get_contact_inds(ind, sim.people.contacts[lkey])
#
                        #self.symPeople[ind].setContacts(contacts)
                        #self.symPeople[ind].selfQuaranteen()

                if counter1 >= new_symptomatic:
                    break;

        #if sum_ns_nc_nd_nr > 0:
#
        #    for ind in self.symPeople.keys():
#
        #        counter2 = counter2 + 1
#
        #        status = 'symptomatic'
#
        #        if sim.people.dead[ind]:
        #            status = 'dead'
        #        elif sim.people.critical[ind]:
        #            status = 'critical'
        #        elif sim.people.severe[ind]:
        #            status = 'severe'
        #        elif sim.people.recovered[ind]:
        #            status = 'recovered'
#
        #        if status != self.symPeople[ind].status:
        #            self.symPeople[ind].onStatusChanged(status)
#
        #        if counter2 >= sum_ns_nc_nd_nr:
        #            break;


class symPerson:

    def __init__(self, uid, sim, layers):
        self.uid = sc.dcp(uid)
        self.contacts = None
        self.cutContacts = dict()
        self.status = 'symptomatic'
        self.sim = sim
        self.layers = sc.dcp(layers)

    def setContacts(self, contactIDs):

        self.contacts = sc.dcp(contactIDs)

        if not iter(contactIDs):
            contactIDs = sc.promotetolist(contactIDs)

        self.contacts = contactIDs

    def selfQuaranteen(self):
        for lkey in self.layers:

            self.cutContacts[lkey] = self.contacts[lkey]
            self.contacts[lkey] = None

            for ind in self.cutContacts[lkey]:
                self.sim.people.contacts[lkey]['beta'][ind] = 0.0

    def onStatusChanged(self, newStatus):
        if self.status == newStatus:
            # debug
            print('called onStatusChanged() but status did not change. This sould never happen. Check code.')
            return

        elif self.status == 'symptomatic' and (newStatus == 'severe' or newStatus == 'critical'):
            self.sym_to_sev_or_cr()

        elif newStatus == 'dead' or newStatus == 'recovered':
            self.resetContacts()  # sim handles dead people as not infectious. Contacts are not relevant anymore.

        self.status = newStatus

    def sym_to_sev_or_cr(self):
        for lkey in self.contacts.keys():
            if self.contacts[lkey] is not None:

                self.cutContacts[lkey] = self.contacts[lkey]
                self.contacts[lkey] = None

                for ind in self.cutContacts[lkey]:
                    self.sim.people.contacts[lkey]['beta'][ind] = 0.0

    def resetContacts(self):
        for lkey in self.cutContacts.keys():
            self.contacts[lkey] = self.cutContacts[lkey]
            self.cutContacts[lkey] = None

            for ind in self.contacts[lkey]:
                self.sim.people.contacts[lkey]['beta'][ind] = 1.0
