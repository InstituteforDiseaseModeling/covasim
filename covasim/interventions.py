import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
import covasim as cv
from . import utils as cvu



__all__ = ['Intervention', 'dynamic_pars', 'sequence', 'change_beta', 'test_num', 'test_prob', 'contact_tracing']


#%% Generic intervention classes

class Intervention:
    '''
    Abstract class for interventions

    '''
    def __init__(self):
        self.results = {}  #: All interventions are guaranteed to have results, so `Sim` can safely iterate over this dict


    def apply(self, sim):
        '''
        Apply intervention

        Function signature matches existing intervention definition
        This method gets called at each timestep and must be implemented
        by derived classes

        Args:
            self:
            sim: The Sim instance

        Returns:
            None
        '''
        raise NotImplementedError


    def plot(self, sim, ax):
        '''
        Call function during plotting

        This can be used to do things like add vertical lines on days when interventions take place

        Args:
            sim: the Sim instance
            ax: the axis instance

        Returns:
            None
        '''
        return


    def to_json(self):
        '''
        Return JSON-compatible representation

        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. In the first instance, the object dict will be returned.
        However, if an intervention itself contains non-standard variables as
        attributes, then its `to_json` method will need to handle those

        Returns:
            JSON-serializable representation (typically a dict, but could be anything else)
        '''
        d = sc.dcp(self.__dict__)
        d['InterventionType'] = self.__class__.__name__
        return d


class dynamic_pars(Intervention):
    '''
    A generic intervention that modifies a set of parameters at specified points
    in time.

    The intervention takes a single argument, pars, which is a dictionary of which
    parameters to change, with following structure: keys are the parameters to change,
    then subkeys 'days' and 'vals' are either a scalar or list of when the change(s)
    should take effect and what the new value should be, respectively.

    Args:
        pars (dict): described above

    Examples:
        interv = cv.dynamic_pars({'diag_factor':{'days':30, 'vals':0.5}, 'cont_factor':{'days':30, 'vals':0.5}}) # Starting day 30, make diagnosed people and people with contacts half as likely to transmit
        interv = cv.dynamic_pars({'beta':{'days':[14, 28], 'vals':[0.005, 0.015]}}) # On day 14, change beta to 0.005, and on day 28 change it back to 0.015
    '''

    def __init__(self, pars):
        super().__init__()
        subkeys = ['days', 'vals']
        for parkey in pars.keys():
            for subkey in subkeys:
                if subkey not in pars[parkey].keys():
                    errormsg = f'Parameter {parkey} is missing subkey {subkey}'
                    raise KeyError(errormsg)
                if sc.isnumber(pars[parkey][subkey]): # Allow scalar values or dicts, but leave everything else unchanged
                    pars[parkey][subkey] = sc.promotetoarray(pars[parkey][subkey])
            len_days = len(pars[parkey]['days'])
            len_vals = len(pars[parkey]['vals'])
            if len_days != len_vals:
                raise ValueError(f'Length of days ({len_days}) does not match length of values ({len_vals}) for parameter {parkey}')
        self.pars = pars
        return


    def apply(self, sim):
        ''' Loop over the parameters, and then loop over the days, applying them if any are found '''
        t = sim.t
        for parkey,parval in self.pars.items():
            inds = sc.findinds(parval['days'], t) # Look for matches
            if len(inds):

                if len(inds)>1:
                    raise ValueError(f'Duplicate days are not allowed for Dynamic interventions (day={t}, indices={inds})')
                else:
                    match = inds[0]
                    val = parval['vals'][match]
                    if isinstance(val, dict):
                        sim[parkey].update(val) # Set the parameter if a nested dict
                    else:
                        sim[parkey] = val # Set the parameter if not a dict
        return


class sequence(Intervention):
    '''
    This is an example of a meta-intervention which switches between a sequence of interventions.

    Args:
        days (list): the days on which to apply each intervention
        interventions (list): the interventions to apply on those days
        WARNING: Will take first intervation after sum(days) days has ellapsed!

    Example:
        interv = cv.sequence(days=[10, 51], interventions=[
                    cv.test_historical(npts, n_tests=[100] * npts, n_positive=[1] * npts),
                    cv.test_prob(npts, symptomatic_prob=0.2, asymptomatic_prob=0.002),
                ])
    '''

    def __init__(self, days, interventions):
        super().__init__()
        assert len(days) == len(interventions)
        self.days = days
        self.interventions = interventions
        self._cum_days = np.cumsum(days)
        return


    def apply(self, sim: cv.Sim):
        idx = np.argmax(self._cum_days > sim.t)  # Index of the intervention to apply on this day
        self.interventions[idx].apply(sim)
        return


class change_beta(Intervention):
    '''
    The most basic intervention -- change beta by a certain amount.

    Args:
        days (int or array): the day or array of days to apply the interventions
        changes (float or array): the changes in beta (1 = no change, 0 = no transmission)
        layers (str or array): the layers in which to change beta

    Examples:
        interv = cv.change_beta(25, 0.3) # On day 25, reduce overall beta by 70% to 0.3
        interv = cv.change_beta([14, 28], [0.7, 1], layers='s') # On day 14, reduce beta by 30%, and on day 28, return to 1 for schools
    '''

    def __init__(self, days, changes, layers=None):
        super().__init__()
        self.days = sc.promotetoarray(days)
        self.changes = sc.promotetoarray(changes)
        self.layers = sc.promotetolist(layers, keepnone=True)
        if len(self.days) != len(self.changes):
            errormsg = f'Number of days supplied ({len(self.days)}) does not match number of changes in beta ({len(self.changes)})'
            raise ValueError(errormsg)
        self.orig_betas = None
        return


    def apply(self, sim):

        # If this is the first time it's being run, store beta
        if self.orig_betas is None:
            self.orig_betas = {}
            for layer in self.layers:
                if layer is None:
                    self.orig_betas['overall'] = sim['beta']
                else:
                    self.orig_betas[layer] = sim['beta_layer'][layer]

        # If this day is found in the list, apply the intervention
        inds = sc.findinds(self.days, sim.t)
        if len(inds):
            for layer,new_beta in self.orig_betas.items():
                for ind in inds:
                    new_beta = new_beta * self.changes[ind]
                if layer == 'overall':
                    sim['beta'] = new_beta
                else:
                    sim['beta_layer'][layer] = new_beta
        return


    def plot(self, sim, ax):
        ''' Plot vertical lines for when changes in beta '''
        ylims = ax.get_ylim()
        for day in self.days:
            pl.plot([day]*2, ylims, '--', c=[0,0,0])
        return


#%% Testing interventions

class test_num(Intervention):
    '''
    Test a fixed number of people per day.

    Example:
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts)

    Returns:
        Intervention
    '''

    def __init__(self, daily_tests, sympt_test=100.0, quar_test=1.0, sensitivity=1.0, test_delay=0, loss_prob=0):
        super().__init__()

        self.daily_tests = daily_tests #: Should be a list of length matching time
        self.sympt_test = sympt_test
        self.quar_test = quar_test
        self.sensitivity = sensitivity
        self.test_delay = test_delay
        self.loss_prob = loss_prob

        return


    def apply(self, sim):

        t = sim.t

        # Process daily tests -- has to be here rather than init so have access to the sim object
        if isinstance(self.daily_tests, (pd.Series, pd.DataFrame)):
            start_date = sim['start_day']
            end_date = self.daily_tests.index[-1]
            dateindex = pd.date_range(start_date, end_date)
            self.daily_tests = self.daily_tests.reindex(dateindex, fill_value=0).to_numpy()

        # Check that there are still tests
        if t < len(self.daily_tests):
            n_tests = self.daily_tests[t]  # Number of tests for this day
            if not (n_tests and pl.isfinite(n_tests)): # If there are no tests today, abort early
                return
            else:
                sim.results['new_tests'][t] += n_tests
        else:
            return

        test_probs = np.ones(sim.n) # Begin by assigning equal tesitng probability to everyone
        symp_inds = cvu.true(sim.people.symptomatic)
        quar_inds = cvu.true(sim.people.quarantined)
        diag_inds = cvu.true(sim.people.diagnosed)
        test_probs[symp_inds] *= self.sympt_test
        test_probs[quar_inds] *= self.quar_test
        test_probs[diag_inds] = 0.

        test_inds = cvu.choose_w(probs=test_probs, n=n_tests, unique=False)

        sim.people.test(test_inds, self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)

        return


class test_prob(Intervention):
    '''
    Test as many people as required based on test probability.
    Probabilities are OR together, so choose wisely.

    Args:
        symptomatic_prob (float): Probability of testing a symptomatic person
        asymptomatic_prob (float): Probability of testing an asymptomatic person
        test_sensitivity (float): Probability of a true positive
        loss_prob (float): Probability of loss to follow-up
        test_delay (int): How long testing takes
        start_day (int): When to start the intervention


    Args:
        symptomatic_prob (float): Probability of testing a symptomatic person
        asymptomatic_prob (float): Probability of testing an asymptomatic person
        test_sensitivity (float): Probability of a true positive
        loss_prob (float): Probability of loss to follow-up
        test_delay (int): How long testing takes
        start_day (int): When to start the intervention


    Example:
        interv = cv.test_prob(symptomatic_prob=0.1, asymptomatic_prob=0.01) # Test 10% of symptomatics and 1% of asymptomatics
        interv = cv.test_prob(symp_quar_prob=0.4) # Test 40% of those in quarantine with symptoms

    Returns:
        Intervention
    '''
    def __init__(self, symp_prob=0, asymp_prob=0, asymp_quar_prob=None, symp_quar_prob=None, test_sensitivity=1.0, loss_prob=0.0, test_delay=1, start_day=0):
        super().__init__()
        self.symp_prob = symp_prob
        self.asymp_prob = asymp_prob
        self.symp_quar_prob = symp_quar_prob if symp_quar_prob is not None else symp_prob
        self.asymp_quar_prob = asymp_quar_prob if asymp_quar_prob is not None else asymp_prob
        self.test_sensitivity = test_sensitivity
        self.loss_prob = loss_prob
        self.test_delay = test_delay
        self.start_day = start_day

        return


    def apply(self, sim):
        ''' Perform testing '''
        t = sim.t
        if t < self.start_day:
            return

        symp_inds       = cvu.true(sim.people.symptomatic)
        asymp_inds      = cvu.false(sim.people.symptomatic)
        quar_inds       = cvu.true(sim.people.quarantined)
        symp_quar_inds  = quar_inds[cvu.true(sim.people.symptomatic[quar_inds])]
        asymp_quar_inds = quar_inds[cvu.false(sim.people.symptomatic[quar_inds])]
        diag_inds       = cvu.true(sim.people.diagnosed)

        test_probs = np.zeros(sim.n) # Begin by assigning equal tesitng probability to everyone
        test_probs[symp_inds]       = self.asymp_prob
        test_probs[asymp_inds]      = self.asymp_prob
        test_probs[symp_quar_inds]  = self.symp_quar_prob
        test_probs[asymp_quar_inds] = self.asymp_quar_prob
        test_probs[diag_inds]       = 0.
        test_inds = cvu.binomial_arr(test_probs).nonzero()[0]

        sim.people.test(test_inds, test_sensitivity=self.test_sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)

        return


class contact_tracing(Intervention):
    '''
    Contact tracing of positives
    '''
    def __init__(self, trace_probs, trace_time, start_day=0, contact_reduction=None):
        super().__init__()
        self.trace_probs = trace_probs
        self.trace_time = trace_time
        self.contact_reduction = contact_reduction # Not using this yet, but could potentially scale contact in this intervention
        self.start_day = start_day
        return

    def apply(self, sim):
        t = sim.t
        if t < self.start_day:
            return

        just_diagnsed_inds = cvu.true(sim.people.diagnosed & (sim.people.date_diagnosed == t-1)) # Diagnosed last time step, time to trace
        if len(just_diagnsed_inds): # If there are any just-diagnosed people, go trace their contacts
            sim.people.trace(just_diagnsed_inds, self.trace_probs, self.trace_time)

        return




# class test_historical(Intervention):
#     '''
#     Test a known number of positive cases

#     This can be used to simulate historical data containing the number of tests performed and the
#     number of cases identified as a result.

#     This intervention will actually test all individuals. At the moment, testing someone who is negative
#     has no effect, so they don't really need to be tested. However, it's possible that in the future
#     a negative test may still have an impact (e.g. make it less likely for an individual to re-test even
#     if they become symptomatic). Therefore to remain as accurate as possible, `Person.test()` is guaranteed
#     to be called for every person tested.

#     One minor limitation of this intervention is that symptomatic individuals that are tested and in reality
#     returned a false negative result would not be tested at all - instead, a non-infectious individual would
#     be tested. At the moment this would not affect model dynamics because a false negative is equivalent to
#     not performing the test at all.
#     '''

#     def __init__(self, n_tests, n_positive):
#         '''
#         Args:
#             n_tests: Number of tests per day. If this is a scalar or an array with length less than npts, it will be zero-padded
#             n_positive: Number of positive tests (confirmed cases) per day. If this is a scalar or an array with length less than npts, it will be zero-padded
#         '''
#         super().__init__()
#         self.n_tests    = sc.promotetoarray(n_tests)
#         self.n_positive = sc.promotetoarray(n_positive)
#         return


#     def apply(self, sim):
#         ''' Perform testing '''

#         t = sim.t

#         if self.n_tests[t]:

#             # Compute weights for people who would test positive or negative
#             positive_tests = np.zeros((sim.n,))
#             for i,person in enumerate(sim.people):
#                 if person.infectious:
#                     positive_tests[i] = 1
#             negative_tests = 1-positive_tests

#             # Select the people to test in each category
#             positive_inds = cv.choose_w(probs=positive_tests, n=min(sum(positive_tests), self.n_positive[t]), normalize=True)
#             negative_inds = cv.choose_w(probs=negative_tests, n=min(sum(negative_tests), self.n_tests[t]-len(positive_inds)), normalize=True)

#             # Todo - assess performance and optimize e.g. to reduce dict indexing
#             for ind in positive_inds:
#                 person = sim.people[ind]
#                 person.test(t, test_sensitivity=1.0) # Sensitivity is 1 because the person is guaranteed to test positive
#                 sim.results['new_diagnoses'][t] += 1

#             for ind in negative_inds:
#                 person = sim.people[ind]
#                 person.test(t, test_sensitivity=1.0)

#             sim.results['new_tests'][t] += self.n_tests[t]

#         return
