import covasim as cv
import pylab as pl
import numpy as np
import sciris as sc

__all__ = ['Intervention', 'dynamic_pars', 'sequence', 'change_beta', 'test_num', 'test_prob', 'test_historical', 'contact_tracing']


#%% Generic intervention classes

class Intervention:
    """
    Abstract class for interventions

    """
    def __init__(self):
        self.results = {}  #: All interventions are guaranteed to have results, so `Sim` can safely iterate over this dict


    def apply(self, sim: cv.Sim) -> None:
        """
        Apply intervention

        Function signature matches existing intervention definition
        This method gets called at each timestep and must be implemented
        by derived classes

        Args:
            self:
            sim: The Sim instance

        Returns:
            None
        """
        raise NotImplementedError


    def plot(self, sim: cv.Sim, ax: pl.Axes) -> None:
        """
        Call function during plotting

        This can be used to do things like add vertical lines on days when interventions take place

        Args:
            sim: the Sim instance
            ax: the axis instance

        Returns:
            None
        """
        return


    def to_json(self):
        """
        Return JSON-compatible representation

        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. In the first instance, the object dict will be returned.
        However, if an intervention itself contains non-standard variables as
        attributes, then its `to_json` method will need to handle those

        Returns: JSON-serializable representation (typically a dict, but could be anything else)

        """
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
                if not sc.isiterable(pars[parkey][subkey]):
                    pars[parkey][subkey] = sc.promotetoarray(pars[parkey][subkey])
            len_days = len(pars[parkey]['days'])
            len_vals = len(pars[parkey]['vals'])
            if len_days != len_vals:
                raise ValueError(f'Length of days ({len_days}) does not match length of values ({len_vals}) for parameter {parkey}')
        self.pars = pars
        return


    def apply(self, sim: cv.Sim):
        ''' Loop over the parameters, and then loop over the days, applying them if any are found '''
        t = sim.t
        for parkey,parval in self.pars.items():
            inds = sc.findinds(parval['days'], t) # Look for matches
            if len(inds):
                if len(inds)>1:
                    raise ValueError(f'Duplicate days are not allowed for Dynamic interventions (day={t}, indices={inds})')
                else:
                    val = parval['vals'][inds[0]]
                    if isinstance(val, dict):
                        sim[parkey].update(val) # Set the parameter if a nested dict
                    else:
                        sim[parkey] = val # Set the parameter if not a dict
        return


class sequence(Intervention):
    """
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
    """

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

    Examples:
        interv = cv.change_beta(25, 0.3) # On day 25, reduce beta by 70% to 0.3
        interv = cv.change_beta([14, 28], [0.7, 1]) # On day 14, reduce beta by 30%, and on day 28, return to 1

    '''

    def __init__(self, days, changes):
        super().__init__()
        self.days = sc.promotetoarray(days)
        self.changes = sc.promotetoarray(changes)
        if len(self.days) != len(self.changes):
            errormsg = f'Number of days supplied ({len(self.days)}) does not match number of changes in beta ({len(self.changes)})'
            raise ValueError(errormsg)
        self.orig_beta = None
        return


    def apply(self, sim: cv.Sim):

        # If this is the first time it's being run, store beta
        if self.orig_beta is None:
            self.orig_beta = sim['beta']

        # If this day is found in the list, apply the intervention
        inds = sc.findinds(self.days, sim.t)
        if len(inds):
            new_beta = self.orig_beta
            for ind in inds:
                new_beta = new_beta * self.changes[ind]
            sim['beta'] = new_beta

        return


    def plot(self, sim: cv.Sim, ax: pl.Axes):
        ''' Plot vertical lines for when changes in beta '''
        ylims = ax.get_ylim()
        for day in self.days:
            pl.plot([day]*2, ylims, '--')
        return


#%% Testing interventions

class test_num(Intervention):
    """
    Test a fixed number of people per day.
    Example:
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts)
    Returns:
        Intervention
    """

    def __init__(self, daily_tests, sympt_test=100.0, quar_test=1.0, sensitivity=1.0, test_delay=0):
        super().__init__()

        self.daily_tests = daily_tests #: Should be a list of length matching time
        self.sympt_test = sympt_test
        self.quar_test = quar_test
        self.sensitivity = sensitivity
        self.test_delay = test_delay

        return


    def apply(self, sim: cv.Sim):

        t = sim.t

        # Check that there are still tests
        if t < len(self.daily_tests):
            n_tests = self.daily_tests[t]  # Number of tests for this day
            sim.results['new_tests'][t] += n_tests
        else:
            return

        # If there are no tests today, abort early
        if not (n_tests and pl.isfinite(n_tests)):
            return

        test_probs = np.ones(sim.n)
        new_diagnoses = 0

        for i, person in enumerate(sim.people.values()):

            new_diagnoses += person.check_diagnosed(t)

            # Adjust testing probability based on what's happened to the person
            # NB, these need to be separate if statements, because a person can be both diagnosed and infectious/symptomatic
            if person.symptomatic:
                test_probs[i] *= self.sympt_test  # They're symptomatic
            if person.quarantine:
                test_probs[i] *= self.quar_test  # They're in quarantine
            if person.diagnosed:
                test_probs[i] = 0.0

        test_inds = cv.choose_weighted(probs=test_probs, n=n_tests, normalize=True)
        sim.results['new_diagnoses'][t] += new_diagnoses

        for test_ind in test_inds:
            person = sim.get_person(test_ind)
            person.test(t, self.sensitivity, test_delay=self.test_delay)

        return


class contact_tracing(Intervention):
    '''
    Contact tracing of positives
    '''
    def __init__(self, trace_probs, trace_time, contact_reduction=None):
        super().__init__()
        self.trace_probs = trace_probs
        self.trace_time = trace_time
        self.contact_reduction = contact_reduction # Not using this yet, but could potentially scale contact in this intervention
        return

    def apply(self, sim: cv.Sim):
        t = sim.t

        # Firstly, loop over diagnosed people to trace their contacts
        diagnosed_ppl = filter(lambda p: p.diagnosed, sim.people.values())
        for i, person in enumerate(diagnosed_ppl):

            # Trace dynamic contact, e.g. the ones that change on every step
            # A sample of community contacts is appended to person.dyn_cont_ppl on each step
            person.trace_dynamic_contacts(self.trace_probs, self.trace_time)

            if person.date_diagnosed is not None and person.date_diagnosed == t-1:
                # This person was just diagnosed: time to trace their (static) contacts
                contactable_ppl = person.trace_static_contacts(self.trace_probs, self.trace_time)
                contactable_ppl.update(person.dyn_cont_ppl)

                # Loop over people who get contacted
                for contact_ind, contact_time in contactable_ppl.items():
                    target_person = sim.get_person(contact_ind)
                    if target_person.date_known_contact is None:
                        target_person.date_known_contact = t + contact_time
                    else:
                        target_person.date_known_contact = min(target_person.date_known_contact, t + contact_time)

        return



class test_prob(Intervention):
    """
    Test as many people as required based on test probability.
    Probabilities are OR together, so choose wisely.

    Example:
        interv = cv.test_prob(symptomatic_prob=0.1, asymptomatic_prob=0.01) # Test 10% of symptomatics and 1% of asymptomatics
        interv = cv.test_prob(symp_quar_prob=0.4) # Test 40% of those in quarantine with symptoms

    Returns:
        Intervention
    """
    def __init__(self, symptomatic_prob=0, asymptomatic_prob=0, quarantine_prob=0, symp_quar_prob=0, test_sensitivity=1.0):
        """

        Args:
            self:
            symptomatic_prob:
            asymptomatic_prob:
            quarantine_prob:
            symp_quar_prob:
            test_sensitivity:

        Returns:

        """
        super().__init__()
        self.symptomatic_prob = symptomatic_prob
        self.asymptomatic_prob = asymptomatic_prob
        self.quarantine_prob = quarantine_prob
        self.symp_quar_prob = symp_quar_prob
        self.test_sensitivity = test_sensitivity
        return


    def apply(self, sim: cv.Sim):
        ''' Perform testing '''
        t = sim.t

        new_tests = 0
        new_diagnoses = 0
        for i, person in enumerate(sim.people.values()):
            new_diagnoses += person.check_diagnosed(t)
            if (person.symptomatic and cv.bt(self.symptomatic_prob)) or \
                (not person.symptomatic and cv.bt(self.asymptomatic_prob)) or \
                (person.quarantine and cv.bt(self.quarantine_prob)) or \
                (person.symptomatic and person.quarantine and cv.bt(self.symp_quar_prob)) :

                new_tests += 1
                person.test(t, self.test_sensitivity)

        sim.results['new_tests'][t] += new_tests
        sim.results['new_diagnoses'][t] += new_diagnoses

        return


class test_historical(Intervention):
    """
    Test a known number of positive cases

    This can be used to simulate historical data containing the number of tests performed and the
    number of cases identified as a result.

    This intervention will actually test all individuals. At the moment, testing someone who is negative
    has no effect, so they don't really need to be tested. However, it's possible that in the future
    a negative test may still have an impact (e.g. make it less likely for an individual to re-test even
    if they become symptomatic). Therefore to remain as accurate as possible, `Person.test()` is guaranteed
    to be called for every person tested.

    One minor limitation of this intervention is that symptomatic individuals that are tested and in reality
    returned a false negative result would not be tested at all - instead, a non-infectious individual would
    be tested. At the moment this would not affect model dynamics because a false negative is equivalent to
    not performing the test at all.

    """

    def __init__(self, n_tests, n_positive):
        """

        Args:
            n_tests: Number of tests per day. If this is a scalar or an array with length less than npts, it will be zero-padded
            n_positive: Number of positive tests (confirmed cases) per day. If this is a scalar or an array with length less than npts, it will be zero-padded
        """
        super().__init__()
        self.n_tests    = sc.promotetoarray(n_tests)
        self.n_positive = sc.promotetoarray(n_positive)
        return


    def apply(self, sim: cv.Sim):
        ''' Perform testing '''

        t = sim.t

        if self.n_tests[t]:

            # Compute weights for people who would test positive or negative
            positive_tests = np.zeros((sim.n,))
            for i, person in enumerate(sim.people.values()):
                if person.infectious:
                    positive_tests[i] = 1
            negative_tests = 1-positive_tests

            # Select the people to test in each category
            positive_inds = cv.choose_weighted(probs=positive_tests, n=min(sum(positive_tests), self.n_positive[t]), normalize=True)
            negative_inds = cv.choose_weighted(probs=negative_tests, n=min(sum(negative_tests), self.n_tests[t]-len(positive_inds)), normalize=True)

            # Todo - assess performance and optimize e.g. to reduce dict indexing
            for ind in positive_inds:
                person = sim.get_person(ind)
                person.test(t, test_sensitivity=1.0) # Sensitivity is 1 because the person is guaranteed to test positive
                sim.results['new_diagnoses'][t] += 1

            for ind in negative_inds:
                person = sim.get_person(ind)
                person.test(t, test_sensitivity=1.0)

            sim.results['new_tests'][t] += self.n_tests[t]

        return
