import covasim as cv
import pylab as pl
import numpy as np

#%% Define classes
class Intervention:
    """
    Abstract class for interventions

    """
    def __init__(self):
        self.results = {}  #: All interventions are guaranteed to have results, so `Sim` can safely iterate over this dict

    def apply(self, sim, t: int) -> None:
        """
        Apply intervention

        Function signature matches existing intervention definition
        This method gets called at each timestep and must be implemented
        by derived classes

        Args:
            self:
            sim: The Sim instance
            t: The current time index

        Returns:

        """
        raise NotImplementedError

    def finalize(self, sim):
        """
        Call function at end of simulation

        This can be used to do things like compute cumulative results

        Args:
            sim:

        Returns:

        """
        return

class ReduceBetaIntervention(Intervention):
    def __init__(self, day, efficacy):
        super().__init__()
        self.day = day
        self.efficacy = efficacy

    def apply(self, sim, t):
        if t == self.day:
            sim['beta'] *= (1-self.efficacy)

class FixedTestIntervention(Intervention):
    """
    Test a fixed number of people per day
    """

    def __init__(self, sim, daily_tests, sympt_test=100.0, trace_test=100.0, sensitivity=1.0):
        super().__init__()

        self.daily_tests = daily_tests #: Should be a list of length matching time
        self.sympt_test = sympt_test
        self.trace_test = trace_test
        self.sensitivity = sensitivity

        self.results['n_diagnoses'] = cv.Result('Number diagnosed', npts=sim.npts)
        self.results['cum_diagnoses'] = cv.Result('Cumulative number diagnosed', npts=sim.npts)

        assert len(self.daily_tests) >= sim.npts, 'Number of daily tests must be specified for at least as many days in the simulation'

    def apply(self, t, sim):

        n_tests = self['daily_tests'][t]  # Number of tests for this day

        # If there are no tests today, abort early
        if not (n_tests and pl.isfinite(n_tests)):
            return

        test_probs = np.ones(sim.npts)

        for i, person in enumerate(sim.people.values()):
            # Adjust testing probability based on what's happened to the person
            # NB, these need to be separate if statements, because a person can be both diagnosed and infectious/symptomatic
            if person.symptomatic:
                test_probs[i] *= self.sympt_test  # They're symptomatic
            if person.known_contact:
                test_probs[i] *= self.trace_test  # They've had contact with a known positive
            if person.diagnosed:
                test_probs[i] = 0.0

        test_probs /= test_probs.sum()
        test_inds = cv.choose_people_weighted(probs=test_probs, n=n_tests)

        for test_ind in test_inds:
            person = sim.get_person(test_ind)
            person.test(t, self.sensitivity)
            if person.diagnosed:
                self.results['n_diagnoses'][t] += 1

    def finalize(self, *args, **kwargs):
        self.results['cum_diagnoses'].values = pl.cumsum(self.results['n_diagnoses'].values)


class FloatingTestIntervention:
    """
    Test as many people as required based on test probability

    Returns:

    """
    def __init__(self, sim, symptomatic_probability=0.5, trace_probability=1.0, test_sensitivity=1.0):
        """

        Args:
            self:
            symptomatic_probability:
            trace_probability:

        Returns:

        """
        super().__init__()
        self.symptomatic_probability = symptomatic_probability
        self.trace_probability = trace_probability # Probability that identified contacts get tested
        self.test_sensitivity = test_sensitivity

        # Instantiate the results to track
        self.results['n_tested'] = cv.Result('Number tested', npts=sim.npts)
        self.results['n_diagnoses'] = cv.Result('Number diagnosed', npts=sim.npts)
        self.results['cum_tested'] = cv.Result('Cumulative number tested', npts=sim.npts)
        self.results['cum_diagnoses'] = cv.Result('Cumulative number diagnosed', npts=sim.npts)

        self.scheduled_tests = set() # Track UIDs of people that are guaranteed to be tested at the next step


    def apply(self, t, sim):
        ''' Perform testing '''

        new_scheduled_tests = set()

        for person in sim.people.values():
            if person.uid in self.scheduled_tests or (person.symptomatic and cv.bt(self.symptomatic_probability)):
                person.test(t, self.test_sensitivity)
                self.results['n_diagnoses'][t] += 1

                for idx in person.contact_inds:
                    if person.diagnosed and self.trace_probability and cv.bt(self.trace_probability):
                        new_scheduled_tests.add(sim.people[idx].uid)

        self.scheduled_tests = new_scheduled_tests


    def finalize(self, *args, **kwargs):
        self.results['cum_tested'].values = pl.cumsum(self.results['n_tested'].values)
        self.results['cum_diagnoses'].values = pl.cumsum(self.results['n_diagnoses'].values)
