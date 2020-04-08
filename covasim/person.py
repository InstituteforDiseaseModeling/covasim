'''
Defines the Person class and functions associated with making people.
'''

#%% Imports
import numpy as np
import sciris as sc
from . import utils as cvu


# Specify all externally visible functions this file defines
__all__ = ['Person']

class Person(sc.prettyobj):
    '''
    Class for a single person.
    '''
    def __init__(self, pars, uid, age, sex, contacts):
        self.uid         = uid # This person's unique identifier
        self.age         = float(age) # Age of the person (in years)
        self.sex         = int(sex) # Female (0) or male (1)
        self.contacts    = contacts # Contacts
        self.durpars     = pars['dur']  # Store duration parameters
        self.dyn_cont_ppl = {} # People who are contactable within the community.  Changes every step so has to be here.

        # Define states -- listed explicitly for performance reasons
        self.susceptible   = True
        self.exposed       = False
        self.infectious    = False
        self.symptomatic   = False
        self.severe        = False
        self.critical      = False
        self.tested        = False
        self.diagnosed     = False
        self.recovered     = False
        self.dead          = False
        self.known_contact = False

        # Define dates
        self.date_exposed       = None
        self.date_infectious    = None
        self.date_symptomatic   = None
        self.date_severe        = None
        self.date_critical      = None
        self.date_tested        = None
        self.date_diagnosed     = None
        self.date_recovered     = None
        self.date_dead          = None
        self.date_known_contact = None

        # Keep track of durations
        self.dur_exp2inf  = None # Duration from exposure to infectiousness
        self.dur_inf2sym  = None # Duration from infectiousness to symptoms
        self.dur_sym2sev  = None # Duration from symptoms to severe symptoms
        self.dur_sev2crit = None # Duration from symptoms to severe symptoms
        self.dur_disease  = None # Total duration of disease, from date of exposure to date of recovery or death

        self.infected = [] #: Record the UIDs of all people this person infected
        self.infected_by = None #: Store the UID of the person who caused the infection. If None but person is infected, then it was an externally seeded infection

        # Set prognoses
        prognoses = pars['prognoses']
        idx = np.argmax(prognoses['age_cutoffs'] > self.age)  # Index of the age bin to use
        self.symp_prob   = pars['rel_symp_prob']   * prognoses['symp_probs'][idx]
        self.severe_prob = pars['rel_severe_prob'] * prognoses['severe_probs'][idx]
        self.crit_prob   = pars['rel_crit_prob']   * prognoses['crit_probs'][idx]
        self.death_prob  = pars['rel_death_prob']  * prognoses['death_probs'][idx]
        self.OR_no_treat = pars['OR_no_treat']

        return


    # Methods to make events occur (infection and diagnosis)
    def infect(self, t, bed_constraint=None, source=None):
        """
        Infect this person and determine their eventual outcomes.
            * Every infected person can infect other people, regardless of whether they develop symptoms
            * Infected people that develop symptoms are disaggregated into mild vs. severe (=requires hospitalization) vs. critical (=requires ICU)
            * Every asymptomatic, mildly symptomatic, and severely symptomatic person recovers
            * Critical cases either recover or die

        Args:
            t: (int) timestep
            bed_constraint: (bool) whether or not there is a bed available for this person
            source: (Person instance), if None, then it was a seed infection

        Returns:
            1 (for incrementing counters)
        """
        self.susceptible    = False
        self.exposed        = True
        self.date_exposed   = t

        # Deal with bed constraint if applicable
        if bed_constraint is None: bed_constraint = False

        # Calculate how long before this person can infect other people
        self.dur_exp2inf     = cvu.sample(**self.durpars['exp2inf'])
        self.date_infectious = t + self.dur_exp2inf

        # Use prognosis probabilities to determine what happens to them
        symp_bool = cvu.bt(self.symp_prob) # Determine if they develop symptoms

        # CASE 1: Asymptomatic: may infect others, but have no symptoms and do not die
        if not symp_bool:  # No symptoms
            dur_asym2rec = cvu.sample(**self.durpars['asym2rec'])
            self.date_recovered = self.date_infectious + dur_asym2rec  # Date they recover
            self.dur_disease = self.dur_exp2inf + dur_asym2rec  # Store how long this person had COVID-19

        # CASE 2: Symptomatic: can either be mild, severe, or critical
        else:
            self.dur_inf2sym = cvu.sample(**self.durpars['inf2sym']) # Store how long this person took to develop symptoms
            self.date_symptomatic = self.date_infectious + self.dur_inf2sym # Date they become symptomatic
            sev_bool = cvu.bt(self.severe_prob) # See if they're a severe or mild case

            # CASE 2a: Mild symptoms, no hospitalization required and no probaility of death
            if not sev_bool: # Easiest outcome is that they're a mild case - set recovery date
                dur_mild2rec = cvu.sample(**self.durpars['mild2rec'])
                self.date_recovered = self.date_symptomatic + dur_mild2rec  # Date they recover
                self.dur_disease = self.dur_exp2inf + self.dur_inf2sym + dur_mild2rec  # Store how long this person had COVID-19

            # CASE 2b: Severe cases: hospitalization required, may become critical
            else:
                self.dur_sym2sev = cvu.sample(**self.durpars['sym2sev']) # Store how long this person took to develop severe symptoms
                self.date_severe = self.date_symptomatic + self.dur_sym2sev  # Date symptoms become severe
                crit_bool = cvu.bt(self.crit_prob)  # See if they're a critical case

                if not crit_bool:  # Not critical - they will recover
                    dur_sev2rec = cvu.sample(**self.durpars['sev2rec'])
                    self.date_recovered = self.date_severe + dur_sev2rec  # Date they recover
                    self.dur_disease = self.dur_exp2inf + self.dur_inf2sym + self.dur_sym2sev + dur_sev2rec  # Store how long this person had COVID-19

                # CASE 2c: Critical cases: ICU required, may die
                else:
                    self.dur_sev2crit = cvu.sample(**self.durpars['sev2crit'])
                    self.date_critical = self.date_severe + self.dur_sev2crit  # Date they become critical
                    this_death_prob = self.death_prob * (self.OR_no_treat if bed_constraint else 1.) # Probability they'll die
                    death_bool = cvu.bt(this_death_prob)  # Death outcome

                    if death_bool:
                        dur_crit2die = cvu.sample(**self.durpars['crit2die'])
                        self.date_dead = self.date_critical + dur_crit2die # Date of death
                        self.dur_disease = self.dur_exp2inf + self.dur_inf2sym + self.dur_sym2sev + self.dur_sev2crit + dur_crit2die   # Store how long this person had COVID-19
                    else:
                        dur_crit2rec = cvu.sample(**self.durpars['crit2rec'])
                        self.date_recovered = self.date_critical + dur_crit2rec # Date they recover
                        self.dur_disease = self.dur_exp2inf + self.dur_inf2sym + self.dur_sym2sev + self.dur_sev2crit + dur_crit2rec  # Store how long this person had COVID-19

        if source:
            self.infected_by = source.uid
            source.infected.append(self.uid)

        return 1 # For incrementing counters


    def trace_dynamic_contacts(self, trace_probs, trace_time, ckey='c'):
        '''
        A method to trace a person's dynamic contacts, e.g. community
        '''
        if ckey in self.contacts:
            this_trace_prob = trace_probs[ckey]
            new_contact_keys = cvu.bf(this_trace_prob, self.contacts[ckey])
            self.dyn_cont_ppl.update({nck:trace_time[ckey] for nck in new_contact_keys})
        return


    def trace_static_contacts(self, trace_probs, trace_time):
        '''
        A method to trace a person's static contacts, e.g. home, school, work
        '''
        contactable_ppl = {}  # Store people that are contactable and how long it takes to contact them
        for ckey in self.contacts.keys():
            if ckey != 'c': # Don't trace community contacts - it's too hard, because they change every timestep
                this_trace_prob = trace_probs[ckey]
                new_contact_keys = cvu.bf(this_trace_prob, self.contacts[ckey])
                contactable_ppl.update({nck: trace_time[ckey] for nck in new_contact_keys})

        return contactable_ppl


    def test(self, t, test_sensitivity, loss_prob=0, test_delay=0):
        '''
        Method to test a person.

        Args:
            t (int): current timestep
            test_sensitivity (float): probability of a true positive
            loss_prob (float): probability of loss to follow-up
            test_delay (int): number of days before test results are ready

        Returns:
            Whether or not this person tested positive
        '''
        self.tested = True

        if self.date_tested is None: # First time tested
            self.date_tested = [t]
        else:
            self.date_tested.append(t) # They're been tested before; append new test date. TODO: adjust testing probs based on whether a person's a repeat tester?

        if self.infectious and cvu.bt(test_sensitivity):  # Person was tested and is true-positive
            needs_diagnosis = not self.date_diagnosed or self.date_diagnosed and self.date_diagnosed > t+test_delay
            if needs_diagnosis and not cvu.bt(loss_prob): # They're not lost to follow-up
                self.date_diagnosed = t + test_delay
            return 1
        else:
            return 0


    # Methods to check a person's status
    def check_symptomatic(self, t):
        ''' Check for new progressions to symptomatic '''
        if not self.symptomatic and self.date_symptomatic and t >= self.date_symptomatic: # Person is changing to this state
            self.symptomatic = True
            return 1
        else:
            return 0


    def check_severe(self, t):
        ''' Check for new progressions to severe '''
        if not self.severe and self.date_severe and t >= self.date_severe: # Person is changing to this state
            self.severe = True
            return 1
        else:
            return 0


    def check_critical(self, t):
        ''' Check for new progressions to critical '''
        if not self.critical and self.date_critical and t >= self.date_critical: # Person is changing to this state
            self.critical = True
            return 1
        else:
            return 0


    def check_recovery(self, t):
        ''' Check if an infected person has recovered '''
        if not self.recovered and self.date_recovered and t >= self.date_recovered: # It's the day they recover
            self.exposed     = False
            self.infectious  = False
            self.symptomatic = False
            self.severe      = False
            self.critical    = False
            self.recovered   = True
            return 1
        else:
            return 0


    def check_death(self, t):
        ''' Check whether or not this person died on this timestep  '''
        if not self.dead and self.date_dead and t >= self.date_dead:
            self.exposed     = False
            self.infectious  = False
            self.symptomatic = False
            self.severe      = False
            self.critical    = False
            self.recovered   = False
            self.dead        = True
            return 1
        else:
            return 0

    def check_diagnosed(self, t):
        ''' Check for new diagnoses '''
        if not self.diagnosed and self.date_diagnosed and t >= self.date_diagnosed: # Person is changing to this state
            self.diagnosed = True
            return 1
        else:
            return 0


