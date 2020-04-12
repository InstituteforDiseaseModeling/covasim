'''
Defines the Person class and functions associated with making people.
'''

#%% Imports
import numpy as np
import sciris as sc
from . import utils as cvu
from . import defaults as cvd


# Specify all externally visible functions this file defines
__all__ = ['People', 'Person']


class People(sc.prettyobj):
    '''
    A class to perform all the operations on the people.
    '''

    def __init__(self, pop_size=None):

        default_dtype = np.float32 # For performance -- 2x faster than float64, the default

        # Set person properties -- mostly floats
        for key in cvd.person_props:
            if key == 'uid':
                self[key] = np.arange(pop_size, dtype=object)
            else:
                self[key] = np.full(pop_size, np.nan, dtype=default_dtype)

        # Set health states -- only susceptible is true by default
        for key in cvd.person_states:
            if key == 'susceptible':
                self[key] = np.full(pop_size, True, dtype=bool)
            else:
                self[key] = np.full(pop_size, False, dtype=bool)

        # Everything else is a float
        for key in cvd.person_dates + cvd.person_durs:
            self[key] = np.full(pop_size, np.nan, dtype=default_dtype)

        return


    def __getitem__(self, key):
        ''' Allow people['attr'] instead of getattr(people, 'attr') '''
        return self.__dict__[key]


    def __setitem__(self, key, value):
        ''' Ditto '''
        self.__dict__[key] = value
        return

    def person(self, ind):
        p = Person()
        for key in cvd.all_person_states:
            setattr(p, key, self[key][ind])
        return p



    # def update(self, t):
    #     ''' Perform all state updates '''

    #     counts = {}

    #     if self.count('severe') > n_beds:
    #         bed_constraint = True

    #     new_infectious  += people.check_infectious(t=t) # For epople who are exposed and not infectious, check if they begin being infectious
    #     new_quarantined += people.check_quar(t=t) # Update if they're quarantined
    #     new_symptomatic += person.check_symptomatic(t)
    #     new_severe      += person.check_severe(t)
    #     new_critical    += person.check_critical(t)
    #     new_deaths      += people.check_death(t=t)
    #     new_recoveries  += person.check_recovery(t)

    #     return counts


    # def update_contacts(self, t):
    #     # Set community contacts

    #             if 'c' in self['contacts']:
    #         n_comm_contacts = self['contacts']['c'] # Community contacts; TODO: make less ugly
    #     else:
    #         n_comm_contacts = 0

    #     person_contacts = person.contacts
    #     if n_comm_contacts:
    #         community_contact_inds = cvu.choose(max_n=pop_size, n=n_comm_contacts)
    #         person_contacts['c'] = community_contact_inds


    # def stuff():
    #     thisbeta = beta * \
    #                (asymp_factor if not person.symptomatic else 1.) * \
    #                (diag_factor if person.diagnosed else 1.)

    #                    this_beta_layer = thisbeta *\
    #                               beta_layers[ckey] *\
    #                               (quar_trans_factor[ckey] if person.quarantined else 1.) # Reduction in onward transmission due to quarantine


    #     # Determine who gets infected
    #     for ckey in self.contact_keys:
    #         contact_ids = person_contacts[ckey]
    #         if len(contact_ids):

    #             transmission_inds = cvu.bf(this_beta_layer, contact_ids)
    #             for contact_ind in transmission_inds: # Loop over people who get infected
    #                 target_person = self.people[contact_ind]
    #                 if target_person.susceptible: # Skip people who are not susceptible

    #                     # See whether we will infect this person
    #                     infect_this_person = True # By default, infect them...
    #                     if target_person.quarantined:
    #                         infect_this_person = cvu.bt(quar_acq_factor) # ... but don't infect them if they're isolating # DJK - should be layer dependent!
    #                     if infect_this_person:
    #                         new_infections += target_person.infect(t, bed_constraint, source=person) # Actually infect them
    #                         sc.printv(f'        Person {person.uid} infected person {target_person.uid}!', 2, verbose)

    # asymp_factor     = self['asymp_factor']
    #     diag_factor      = self['diag_factor']
    #     quar_trans_factor= self['quar_trans_factor']
    #     quar_acq_factor  = self['quar_acq_factor']
    #     quar_period      = self['quar_period']
    #     beta_layers      = self['beta_layers']
    #     n_beds           = self['n_beds']
    #     bed_constraint   = False



class Person(sc.prettyobj):
    '''
    Class for a single person.
    '''
    def __init__(self, pars=None, uid=None, age=-1, sex=-1, contacts=None):
        self.uid         = uid # This person's unique identifier
        self.age         = float(age) # Age of the person (in years)
        self.sex         = int(sex) # Female (0) or male (1)
        self.contacts    = contacts # Contacts
        self.dyn_cont_ppl = {} # People who are contactable within the community.  Changes every step so has to be here.

        # Set states
        self.infected = [] #: Record the UIDs of all people this person infected
        self.infected_by = None #: Store the UID of the person who caused the infection. If None but person is infected, then it was an externally seeded infection

        # Set prognoses
        if pars:
            self.durpars = pars['dur']  # Store duration parameters
            prognoses = pars['prognoses']
            idx = np.argmax(prognoses['age_cutoffs'] > self.age)  # Index of the age bin to use
            self.symp_prob   = pars['rel_symp_prob']   * prognoses['symp_probs'][idx]
            self.severe_prob = pars['rel_severe_prob'] * prognoses['severe_probs'][idx]
            self.crit_prob   = pars['rel_crit_prob']   * prognoses['crit_probs'][idx]
            self.death_prob  = pars['rel_death_prob']  * prognoses['death_probs'][idx]
            self.OR_no_treat = pars['OR_no_treat']

        return


    def make_susceptible(self):
        """
        Make person susceptible. This is used during initialization and dynamic resampling
        """

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
                these_contacts = self.contacts[ckey]
                if len(these_contacts):
                    this_trace_prob = trace_probs[ckey]
                    new_contact_keys = cvu.bf(this_trace_prob, these_contacts)
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


    def quarantine(self, t, quar_period):
        '''
        Quarantine a person starting on day t
        If a person is already quarantined, this will extend their quarantine
        '''
        self.quarantined = True

        new_end_quarantine = t + quar_period
        if self.end_quarantine is None or self.end_quarantine is not None and new_end_quarantine > self.end_quarantine:
            self.end_quarantine = new_end_quarantine

        #sc.printv(f'Person {self.uid} has been quarantined until {self.end_quarantine}', 2, self.verbose)

        return

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


    def check_quar_begin(self, t, quar_period=None):
        ''' Check for whether someone has been contacted by a positive'''
        if (quar_period is not None) and (self.date_known_contact is not None) and (t >= self.date_known_contact):
            # Begin quarantine
            was_quarantined = self.quarantined
            self.quarantine(t, quar_period)
            self.date_known_contact = None # Clear
            return not was_quarantined
        return 0


    def check_quar_end(self, t):
        ''' Check for whether someone is isolating/quarantined'''
        if self.quarantined and (self.end_quarantine is not None) and (t >= self.end_quarantine):
            self.quarantined = False # Release from quarantine
            self.end_quarantine = None # Clear end quarantine time
            #sc.printv(f'Released {self.uid} from quarantine', 2, verbose)
        return self.quarantined
