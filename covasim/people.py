'''
Defines the Person class and functions associated with making people.
'''

#%% Imports
import numpy as np
from . import utils as cvu
from . import defaults as cvd
from . import base as cvb


# Specify all externally visible functions this file defines
__all__ = ['People']



class People(cvb.BasePeople):
    '''
    A class to perform all the operations on the people.
    '''

    def __init__(self, pars=None, pop_size=None, **kwargs):
        super().__init__(pars, pop_size)

        # Set person properties -- mostly floats
        for key in self.keylist.person:
            if key == 'uid':
                self[key] = np.arange(self.pop_size, dtype=object)
            else:
                self[key] = np.full(self.pop_size, np.nan, dtype=self._default_dtype)

        # Set health states -- only susceptible is true by default -- booleans
        for key in self.keylist.states:
            if key == 'susceptible':
                self[key] = np.full(self.pop_size, True, dtype=bool)
            else:
                self[key] = np.full(self.pop_size, False, dtype=bool)

        # Set dates and durations -- both floats
        for key in self.keylist.dates + self.keylist.durs:
            self[key] = np.full(self.pop_size, np.nan, dtype=self._default_dtype)

        # Store the dtypes used
        self._dtypes = {key:self[key].dtype for key in self.keys()} # Assign all to float by default
        self._lock = True # Stop further attributes from being set

        # Set any values, if supplied
        if 'contacts' in kwargs:
            self.add_contacts(kwargs.pop('contacts'))
        for key,value in kwargs.items():
            self.set(key, value)

        return


    def set_prognoses(self, pars=None):
        ''' Set the prognoses for each person based on age '''
        if pars is None:
            pars = self.pars

        def find_cutoff(age_cutoffs, age):
            return np.argmax(age_cutoffs > age)  # Index of the age bin to use

        prognoses = pars['prognoses']
        age_cutoffs = prognoses['age_cutoffs']
        inds = np.fromiter((find_cutoff(age_cutoffs, this_age) for this_age in self.age), dtype=np.int32, count=len(self))
        self.symp_prob   = self.pars['rel_symp_prob']   * prognoses['symp_probs'][inds]
        self.severe_prob = pars['rel_severe_prob'] * prognoses['severe_probs'][inds]
        self.crit_prob   = pars['rel_crit_prob']   * prognoses['crit_probs'][inds]
        self.death_prob  = pars['rel_death_prob']  * prognoses['death_probs'][inds]
        return


    def update_states(self, t):
        ''' Perform all state updates '''

        counts = {key:0 for key in cvd.results_flows}

        # if self.count('severe') > n_beds:
        #     bed_constraint = True

        # new_infectious  += people.check_infectious(t=t) # For epople who are exposed and not infectious, check if they begin being infectious
        # new_quarantined += people.check_quar(t=t) # Update if they're quarantined
        # new_symptomatic += person.check_symptomatic(t)
        # new_severe      += person.check_severe(t)
        # new_critical    += person.check_critical(t)
        # new_deaths      += people.check_death(t=t)
        # new_recoveries  += person.check_recovery(t)

        return counts


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




    def make_susceptible(self, inds):
        """
        Make person susceptible. This is used during dynamic resampling
        """
        for key in self.keylist.states:
            if key == 'susceptible':
                self[key][inds] = True
            else:
                self[key][inds] = False

        for key in self.keylist.dates + self.keylist.durs:
            self[key][inds] = None

        return


    # Methods to make events occur (infection and diagnosis)
    def infect(self, inds, t, bed_constraint=None, source=None):
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
        n_inds = len(inds)
        durpars = self.pars['dur']
        self.susceptible[inds]    = False
        self.exposed[inds]        = True
        self.date_exposed[inds]   = t

        # Deal with bed constraint if applicable
        if bed_constraint is None: bed_constraint = False

        # Calculate how long before this person can infect other people
        self.dur_exp2inf[inds]     = cvu.sample(**durpars['exp2inf'], size=n_inds)
        self.date_infectious[inds] = self.dur_exp2inf[inds] + t

        # Use prognosis probabilities to determine what happens to them
        is_symp = cvu.binomial_arr(self.symp_prob[inds]) # Determine if they develop symptoms
        symp_inds = inds[is_symp]
        asymp_inds = inds[~is_symp] # Asymptomatic

        # CASE 1: Asymptomatic: may infect others, but have no symptoms and do not die
        dur_asym2rec = cvu.sample(**durpars['asym2rec'], size=len(asymp_inds))
        self.date_recovered[asymp_inds] = self.date_infectious[asymp_inds] + dur_asym2rec  # Date they recover
        self.dur_disease[asymp_inds] = self.dur_exp2inf[asymp_inds] + dur_asym2rec  # Store how long this person had COVID-19

        # CASE 2: Symptomatic: can either be mild, severe, or critical
        n_symp_inds = len(symp_inds)
        self.dur_inf2sym[symp_inds] = cvu.sample(**durpars['inf2sym'], size=n_symp_inds) # Store how long this person took to develop symptoms
        self.date_symptomatic[symp_inds] = self.date_infectious[symp_inds] + self.dur_inf2sym[symp_inds] # Date they become symptomatic
        is_sev = cvu.binomial_arr(self.severe_prob[symp_inds]) # See if they're a severe or mild case
        sev_inds = symp_inds[is_sev]
        mild_inds = symp_inds[~is_sev] # Not severe

        # CASE 2a: Mild symptoms, no hospitalization required and no probaility of death
        dur_mild2rec = cvu.sample(**durpars['mild2rec'], size=len(mild_inds))
        self.date_recovered[mild_inds] = self.date_symptomatic[mild_inds] + dur_mild2rec  # Date they recover
        self.dur_disease[mild_inds] = self.dur_exp2inf[mild_inds] + self.dur_inf2sym[mild_inds] + dur_mild2rec  # Store how long this person had COVID-19

        # CASE 2b: Severe cases: hospitalization required, may become critical
        self.dur_sym2sev[sev_inds] = cvu.sample(**durpars['sym2sev'], size=len(sev_inds)) # Store how long this person took to develop severe symptoms
        self.date_severe[sev_inds] = self.date_symptomatic[sev_inds] + self.dur_sym2sev[sev_inds]  # Date symptoms become severe
        is_crit = cvu.binomial_arr(self.crit_prob[sev_inds])  # See if they're a critical case
        crit_inds = sev_inds[is_crit]
        non_crit_inds = sev_inds[~is_crit]

        # Not critical - they will recover
        dur_sev2rec = cvu.sample(**durpars['sev2rec'], size=len(non_crit_inds))
        self.date_recovered[non_crit_inds] = self.date_severe[non_crit_inds] + dur_sev2rec  # Date they recover
        self.dur_disease[non_crit_inds] = self.dur_exp2inf[non_crit_inds] + self.dur_inf2sym[non_crit_inds] + self.dur_sym2sev[non_crit_inds] + dur_sev2rec  # Store how long this person had COVID-19

        # CASE 2c: Critical cases: ICU required, may die
        self.dur_sev2crit[crit_inds] = cvu.sample(**durpars['sev2crit'], size=len(crit_inds))
        self.date_critical[crit_inds] = self.date_severe[crit_inds] + self.dur_sev2crit[crit_inds]  # Date they become critical
        this_death_prob = self.death_prob[crit_inds] * (self.pars['OR_no_treat'] if bed_constraint else 1.) # Probability they'll die
        is_dead = cvu.binomial_arr(this_death_prob)  # Death outcome
        death_inds = crit_inds[is_dead]
        alive_inds = crit_inds[~is_dead]

        # Did not die
        dur_crit2rec = cvu.sample(**durpars['crit2rec'], size=len(alive_inds))
        self.date_recovered[alive_inds] = self.date_critical[alive_inds] + dur_crit2rec # Date they recover
        self.dur_disease[alive_inds] = self.dur_exp2inf[alive_inds] + self.dur_inf2sym[alive_inds] + self.dur_sym2sev[alive_inds] + self.dur_sev2crit[alive_inds] + dur_crit2rec  # Store how long this person had COVID-19

        # Did die
        dur_crit2die = cvu.sample(**durpars['crit2die'], size=len(death_inds))
        self.date_dead[death_inds] = self.date_critical[death_inds] + dur_crit2die # Date of death
        self.dur_disease[death_inds] = self.dur_exp2inf[death_inds] + self.dur_inf2sym[death_inds] + self.dur_sym2sev[death_inds] + self.dur_sev2crit[death_inds] + dur_crit2die   # Store how long this person had COVID-19

        # if source:
        #     self.infected_by = source.uid
        #     source.infected.append(self.uid)

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
