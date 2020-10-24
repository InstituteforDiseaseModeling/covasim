import sciris as sc
from covasim import base as cvb
from covasim import Intervention
from covasim.interventions import process_changes, process_days, find_day

def process_keys(sim, keys):

    if sc.isstring(keys) or not sc.isiterable(keys):
        keys = sc.promotetolist(keys)

    for key in keys:

        watchable_keys = sc.promotetolist()

        for key in sim.result_keys():
            if 'n_' in key or 'new_' in key:
                watchable_keys.append(key)

        if key not in sim.result_keys():
            errormsg = f'key "{key}" does not match any watchable value. Watchable values: {str(watchable_keys)[1:-1]}'
            raise KeyError(errormsg)

    return keys


def process_thresholds(thresholds):

    if not sc.isiterable(thresholds):
        thresholds = sc.promotetolist(thresholds)

    for threshold in thresholds:
        if not isinstance(threshold, int ):
            errormsg = f'Only ints are allowed for thresholds. Type recognized: {type(threshold)}'
            raise TypeError(errormsg)

    return thresholds


def process_interventions(interventions):

    if not sc.isiterable(interventions):
        interventions = sc.promotetolist(interventions)

    for intervention in interventions:
        if not isinstance(intervention, Intervention):
            errormsg = f'Intervention has to be of type "Intervention". Type recognized: {type(intervention)}'
            raise TypeError(errormsg)

    return interventions


class threshold_intervention(Intervention):
    '''

    Args:
        key (string or array): given key value(s) to be observed
        threshold (int or array): value on witch a given intervention should take place
        intervention_th_exceeded (Intervention or array): intervention(s) to be applied for exceeded threshold
        intervention_th_underrun (Intervention or array): intervention(s) to be applied for underrun threshold
        kwargs (dict): passed to Intervention()

    **Example**::

        intervention_over_th = cv.change_beta(0, 0.3)
        intervention_under_th = cv.change_beta(0, 1)

        th_intervention = threshold_intervention('n_severe', 20, intervention_over_th, intervention_under_th)

    '''

    def __init__(self, keys, thresholds, intervention_th_exceeded, intervention_th_underrun,**kwargs):
        super().__init__(**kwargs)  # Initialize the Intervention object
        self._store_args()  # Store the input arguments so the intervention can be recreated
        self.keys = sc.dcp(keys)
        self.thresholds = sc.dcp(thresholds)
        self.intervention_th_exceeded = sc.dcp(intervention_th_exceeded)
        self.intervention_th_underrun = sc.dcp(intervention_th_underrun)
        return

    def initialize(self, sim):
        self.days = sc.promotetolist()
        self.keys = process_keys(sim, self.keys)
        self.thresholds = process_thresholds(self.thresholds)
        self.intervention_th_exceeded = process_interventions(self.intervention_th_exceeded)
        self.intervention_th_underrun = process_interventions(self.intervention_th_underrun)
        #self.contacts = cvb.Contacts(layer_keys=sim.layer_keys())  # all Layers
        self.exceeded = [0] * len(self.thresholds)
        self.initialized = True
        return

    def apply(self, sim):

        daybefore = max(sim.t-1,0)
        day = sim.t

        for i, key in enumerate(self.keys):

            if sim.results[key][daybefore] >= self.thresholds[i] and not self.exceeded[i]:

                self.exceeded[i] = 1

                self.days.append(day)
                self.intervention_th_exceeded[i].days = sc.promotetolist(day)
                self.intervention_th_exceeded[i].input_args['days'] = sc.promotetolist(day)
                self.intervention_th_exceeded[i].lable = f'{key} on day {day}'

                if not self.intervention_th_exceeded[i].initialized:
                    self.intervention_th_exceeded[i].initialize(sim)

                self.intervention_th_exceeded[i].apply(sim)

                #debug
                #print(f'exceed {key}: {sim.results[key][daybefore]}/{self.thresholds[i]}')

            elif sim.results[key][daybefore] < self.thresholds[i] and self.exceeded[i]:

                self.exceeded[i] = 0

                self.days.append(day)
                self.intervention_th_underrun[i].days = sc.promotetolist(day)
                self.intervention_th_underrun[i].input_args['days'] = sc.promotetolist(day)
                self.intervention_th_underrun[i].lable = f'{key} on day {day}'

                if not self.intervention_th_underrun[i].initialized:
                    self.intervention_th_underrun[i].initialize(sim)

                self.intervention_th_underrun[i].apply(sim)

                #debug
                #print(f'underrun {key}: {sim.results[key][daybefore]}/{self.thresholds[i]}')

        return
