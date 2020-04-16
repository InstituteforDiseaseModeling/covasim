'''
Defines the Person class and functions associated with making people.
'''

#%% Imports
import numpy as np
import sciris as sc
from . import utils as cvu
from . import defaults as cvd
from . import base as cvb


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


    def initialize(self, pars=None, dynamic_keys=None):
        ''' Perform initializations '''
        self.set_prognoses(pars)
        self.set_betas(pars)
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
        self.symp_prob[:]   = pars['rel_symp_prob']   * prognoses['symp_probs'][inds]
        self.severe_prob[:] = pars['rel_severe_prob'] * prognoses['severe_probs'][inds]
        self.crit_prob[:]   = pars['rel_crit_prob']   * prognoses['crit_probs'][inds]
        self.death_prob[:]  = pars['rel_death_prob']  * prognoses['death_probs'][inds]
        self.rel_sus[:]     = 1.0 # By default: is susceptible
        self.rel_trans[:]   = 0.0 # By default: cannot transmit

        return


    def set_betas(self, pars=None):
        ''' Set betas for each layer '''
        if pars is None:
            pars = self.pars


        for key,value in pars['beta_layer'].items():
            df = self.contacts[key]
            df['beta'][:] = value

        return


    def update_states(self, t):
        ''' Perform all state updates '''

        # Initialize
        self.t = t
        counts = {key:0 for key in cvd.new_result_flows}
        self.is_exp = cvu.true(self.exposed) # For storing the interim values since used in every subsequent calculation

        # Perform updates
        counts['new_infectious']  += self.check_infectious() # For people who are exposed and not infectious, check if they begin being infectious
        counts['new_symptomatic'] += self.check_symptomatic()
        counts['new_severe']      += self.check_severe()
        counts['new_critical']    += self.check_critical()
        counts['new_deaths']      += self.check_death()
        counts['new_recoveries']  += self.check_recovery()
        counts['new_quarantined'] += self.check_quar() # Update if they're quarantined
        counts['new_diagnoses']   += self.check_diagnosed()
        counts['new_tests']       += self.check_tested()
        del self.is_exp # Tidy up

        return counts


    def update_contacts(self, dynamic_keys='c'):
        ''' Set dynamic contacts, by default, community ('c') '''

        # Remove existing dynamic contacts
        self.remove_dynamic_contacts()

        # Figure out if anything needs to be done
        dynamic_keys = sc.promotetolist(dynamic_keys)
        for dynamic_key in dynamic_keys:
            if dynamic_key in self.layer_keys():
                pop_size   = len(self)
                n_contacts = self.pars['contacts'][dynamic_key]
                beta       = self.pars['beta_layer'][dynamic_key]

                # Create new contacts
                n_new = n_contacts*pop_size
                new_contacts = {} # Initialize
                new_contacts['p1'] = np.array(cvu.choose_r(max_n=pop_size, n=n_new), dtype=np.int32)
                new_contacts['p2'] = np.array(cvu.choose_r(max_n=pop_size, n=n_new), dtype=np.int32)

                # Set the things for the entire list
                new_contacts['layer'] = np.array([dynamic_key]*n_new)
                new_contacts['beta']  = np.array([beta]*n_new, dtype=np.float32)

                # Add to contacts
                self.add_contacts(new_contacts, key=dynamic_key)
                self.contacts[dynamic_key].validate()

        return self.contacts


    def make_susceptible(self, inds):
        '''
        Make person susceptible. This is used during dynamic resampling
        '''
        for key in self.keylist.states:
            if key == 'susceptible':
                self[key][inds] = True
            else:
                self[key][inds] = False

        for key in self.keylist.dates + self.keylist.durs:
            self[key][inds] = np.nan

        return


    #%% Methods for updating state

    def check_inds(self, current, date, filter_inds=None):
        ''' Return indices for which the current state is false nad which meet the date criterion '''
        if filter_inds is None:
            filter_inds = self.is_exp
        not_current = cvu.ifalsei(current, filter_inds)
        has_date    = cvu.idefinedi(date, not_current)
        inds        = cvu.itrue(self.t >= date[has_date], has_date)
        return inds


    def check_infectious(self):
        ''' Check if they become infectious '''
        inds = self.check_inds(self.infectious, self.date_infectious)
        self.infectious[inds] = True
        self.rel_trans[inds]  = 1.0 # TODO: make this dynamic
        return len(inds)


    def check_symptomatic(self):
        ''' Check for new progressions to symptomatic '''
        inds = self.check_inds(self.symptomatic, self.date_symptomatic)
        self.symptomatic[inds] = True
        return len(inds)


    def check_severe(self):
        ''' Check for new progressions to severe '''
        inds = self.check_inds(self.severe, self.date_severe)
        self.severe[inds] = True
        return len(inds)


    def check_critical(self):
        ''' Check for new progressions to critical '''
        inds = self.check_inds(self.critical, self.date_critical)
        self.critical[inds] = True
        return len(inds)


    def check_recovery(self):
        ''' Check for recovery '''
        inds = self.check_inds(self.recovered, self.date_recovered)
        self.exposed[inds]     = False
        self.infectious[inds]  = False
        self.symptomatic[inds] = False
        self.severe[inds]      = False
        self.critical[inds]    = False
        self.recovered[inds]   = True
        self.rel_trans[inds]   = 0.0
        return len(inds)


    def check_death(self):
        ''' Check whether or not this person died on this timestep  '''
        inds = self.check_inds(self.dead, self.date_dead)
        self.exposed[inds]     = False
        self.infectious[inds]  = False
        self.symptomatic[inds] = False
        self.severe[inds]      = False
        self.critical[inds]    = False
        self.recovered[inds]   = False
        self.dead[inds]        = True
        self.rel_trans[inds]   = 0.0
        return len(inds)


    def check_tested(self):
        ''' Check for new tests '''
        inds = self.check_inds(self.tested, self.date_tested)
        self.tested[inds] = True
        return len(inds)


    def check_diagnosed(self):
        ''' Check for new diagnoses '''
        inds = self.check_inds(self.diagnosed, self.date_diagnosed)
        self.diagnosed[inds] = True
        return len(inds)


    def check_quar(self):
        ''' Check for whether someone has been contacted by a positive'''

        if self.pars['quar_period'] is not None:

            # Perform quarantine
            inds = self.check_inds(self.quarantined, self.date_known_contact) # Check who is quarantined
            self.quarantine(inds) # Put people in quarantine
            self.date_known_contact[inds] = np.nan # Clear date

            # Check for the end of quarantine
            end_inds = self.check_inds(~self.quarantined, self.date_end_quarantine) # Note the double-negative here
            self.quarantined[end_inds] = False # Release from quarantine
            self.date_end_quarantine[end_inds] = np.nan # Clear end quarantine time

            n_quarantined = len(inds)

        else:
            n_quarantined = 0

        return n_quarantined


    #%% Methods to make events occur (infection and diagnosis)

    def infect(self, inds, bed_max=None, verbose=True):
        '''
        Infect people and determine their eventual outcomes.
            * Every infected person can infect other people, regardless of whether they develop symptoms
            * Infected people that develop symptoms are disaggregated into mild vs. severe (=requires hospitalization) vs. critical (=requires ICU)
            * Every asymptomatic, mildly symptomatic, and severely symptomatic person recovers
            * Critical cases either recover or die

        Args:
            inds    (array):  array of people to infect
            t       (int):    current timestep
            bed_max (bool):   whether or not there is a bed available for this person

        Returns:
            count (int): number of people infected
        '''

        # Handle inputs
        n_infections = len(inds)
        durpars      = self.pars['dur']

        # Set states
        self.susceptible[inds]    = False
        self.exposed[inds]        = True
        self.rel_sus[inds]        = 0.0 # Not susceptible after becoming infected
        self.date_exposed[inds]   = self.t

        # Deal with bed constraint if applicable
        if bed_max is None: bed_max = False

        # Calculate how long before this person can infect other people
        self.dur_exp2inf[inds]     = cvu.sample(**durpars['exp2inf'], size=n_infections)
        self.date_infectious[inds] = self.dur_exp2inf[inds] + self.t

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

        # CASE 2.1: Mild symptoms, no hospitalization required and no probaility of death
        dur_mild2rec = cvu.sample(**durpars['mild2rec'], size=len(mild_inds))
        self.date_recovered[mild_inds] = self.date_symptomatic[mild_inds] + dur_mild2rec  # Date they recover
        self.dur_disease[mild_inds] = self.dur_exp2inf[mild_inds] + self.dur_inf2sym[mild_inds] + dur_mild2rec  # Store how long this person had COVID-19

        # CASE 2.2: Severe cases: hospitalization required, may become critical
        self.dur_sym2sev[sev_inds] = cvu.sample(**durpars['sym2sev'], size=len(sev_inds)) # Store how long this person took to develop severe symptoms
        self.date_severe[sev_inds] = self.date_symptomatic[sev_inds] + self.dur_sym2sev[sev_inds]  # Date symptoms become severe
        is_crit = cvu.binomial_arr(self.crit_prob[sev_inds])  # See if they're a critical case
        crit_inds = sev_inds[is_crit]
        non_crit_inds = sev_inds[~is_crit]

        # CASE 2.2.1 Not critical - they will recover
        dur_sev2rec = cvu.sample(**durpars['sev2rec'], size=len(non_crit_inds))
        self.date_recovered[non_crit_inds] = self.date_severe[non_crit_inds] + dur_sev2rec  # Date they recover
        self.dur_disease[non_crit_inds] = self.dur_exp2inf[non_crit_inds] + self.dur_inf2sym[non_crit_inds] + self.dur_sym2sev[non_crit_inds] + dur_sev2rec  # Store how long this person had COVID-19

        # CASE 2.2.2: Critical cases: ICU required, may die
        self.dur_sev2crit[crit_inds] = cvu.sample(**durpars['sev2crit'], size=len(crit_inds))
        self.date_critical[crit_inds] = self.date_severe[crit_inds] + self.dur_sev2crit[crit_inds]  # Date they become critical
        this_death_prob = self.death_prob[crit_inds] * (self.pars['OR_no_treat'] if bed_max else 1.) # Probability they'll die
        is_dead = cvu.binomial_arr(this_death_prob)  # Death outcome
        dead_inds = crit_inds[is_dead]
        alive_inds = crit_inds[~is_dead]

        # CASE 2.2.2.1: Did not die
        dur_crit2rec = cvu.sample(**durpars['crit2rec'], size=len(alive_inds))
        self.date_recovered[alive_inds] = self.date_critical[alive_inds] + dur_crit2rec # Date they recover
        self.dur_disease[alive_inds] = self.dur_exp2inf[alive_inds] + self.dur_inf2sym[alive_inds] + self.dur_sym2sev[alive_inds] + self.dur_sev2crit[alive_inds] + dur_crit2rec  # Store how long this person had COVID-19

        # CASE 2.2.2.2: Did die
        dur_crit2die = cvu.sample(**durpars['crit2die'], size=len(dead_inds))
        self.date_dead[dead_inds] = self.date_critical[dead_inds] + dur_crit2die # Date of death
        self.dur_disease[dead_inds] = self.dur_exp2inf[dead_inds] + self.dur_inf2sym[dead_inds] + self.dur_sym2sev[dead_inds] + self.dur_sev2crit[dead_inds] + dur_crit2die   # Store how long this person had COVID-19

        return n_infections # For incrementing counters


    def test(self, inds, test_sensitivity=1.0, loss_prob=0.0, test_delay=0):
        '''
        Method to test people

        Args:
            inds: indices of who to test
            test_sensitivity (float): probability of a true positive
            loss_prob (float): probability of loss to follow-up
            test_delay (int): number of days before test results are ready

        Returns:
            Whether or not this person tested positive
        '''
        self.tested[inds] = True
        self.date_tested[inds] = self.t # Only keep the last time they tested

        is_infectious = cvu.itruei(self.infectious, inds)
        pos_test      = cvu.n_binomial(test_sensitivity, len(is_infectious))
        is_inf_pos    = is_infectious[pos_test]

        not_diagnosed = is_inf_pos[np.isnan(self.date_diagnosed[is_inf_pos])]
        not_lost      = cvu.n_binomial(test_sensitivity, len(not_diagnosed))
        inds          = not_diagnosed[not_lost]

        self.date_diagnosed[inds] = self.t + test_delay

        return


    def quarantine(self, inds):
        '''
        Quarantine selected people starting on the current day. If a person is already
        quarantined, this will extend their quarantine.
        Args:
            inds (array): indices of who to quarantine
        '''
        self.quarantined[inds] = True
        self.date_end_quarantine[inds] = self.t + self.pars['quar_period']
        return


    def trace(self, inds, trace_probs, trace_time):
        '''
        Trace the contacts of the people provided
        Args:
            inds (array): indices of whose contacts to trace
            trace_probs (dict): probability of being able to trace people at each contact layer - should have the same keys as contacts
            trace_time (dict): # days it'll take to trace people at each contact layer - should have the same keys as contacts
        '''
        # Figure out who has been contacted in the past
        never_been_contacted = cvu.true(np.isnan(self.date_known_contact))  # Indices of people who've never been contacted

        # Extract the indices of the people who'll be contacted
        for layer in self.layer_keys():
            this_trace_prob = trace_probs[layer]
            this_trace_time = trace_time[layer]

            p1inds = np.where(np.isin(np.array(self.contacts[layer]['p1']),inds))[0] # Get all the indices of the pairs that each person is in
            p2inds = np.unique(np.array(self.contacts[layer]['p2'][p1inds])) # Find their pairing partner
            contact_inds = cvu.binomial_filter(this_trace_prob, p2inds) # Filter the indices according to the probability of being able to trace this layer
            self.known_contact[contact_inds] = True

            # Set the date of contact, careful not to override what might be an earlier date. TODO: this could surely be one operation?
            first_time_contacted_inds   = np.intersect1d(never_been_contacted, contact_inds) # indices of people getting contacted for the first time
            contacted_before_inds       = np.setdiff1d(contact_inds, first_time_contacted_inds) # indices of people who've been contacted before

            if len(first_time_contacted_inds):
                self.date_known_contact[first_time_contacted_inds]  = self.t + this_trace_time # Store when they were contacted
            if len(contacted_before_inds):
                self.date_known_contact[contacted_before_inds]  = np.minimum(self.date_known_contact[contacted_before_inds], self.t + this_trace_time)

        return



