'''
Defines the Person class and functions associated with making people.
'''

#%% Imports
import numpy as np
from . import utils as cvu
from . import defaults as cvd
from . import base as cvb
from . import plotting as cvplt


__all__ = ['People']


class People(cvb.BasePeople):
    '''
    A class to perform all the operations on the people. This class is usually
    not invoked directly, but instead is created automatically by the sim. Most
    initialization happens in BasePeople.

    Args:
        pars (dict): the sim parameters, e.g. sim.pars -- must have pop_size and n_days keys
        strict (bool): whether or not to only create keys that are already in self.meta.person; otherwise, let any key be set
        kwargs (dict): the actual data, e.g. from a popdict, being specified
    '''

    def __init__(self, pars, strict=True, **kwargs):
        super().__init__(pars)

        # Handle contacts, if supplied (note: they usually are)
        if 'contacts' in kwargs:
            self.add_contacts(kwargs.pop('contacts'))

        # Handle all other values, e.g. age
        for key,value in kwargs.items():
            if strict:
                self.set(key, value)
            else:
                self[key] = value

        return


    def initialize(self):
        ''' Perform initializations '''
        self.set_prognoses()
        self.validate()
        self.initialized = True
        return


    def set_prognoses(self):
        '''
        Set the prognoses for each person based on age during initialization. Need
        to reset the seed because viral loads are drawn stochastically.
        '''

        pars = self.pars # Shorten

        def find_cutoff(age_cutoffs, age):
            '''
            Find which age bin each person belongs to -- e.g. with standard
            age bins 0, 10, 20, etc., ages [5, 12, 4, 58] would be mapped to
            indices [0, 1, 0, 5]. Age bins are not guaranteed to be uniform
            width, which is why this can't be done as an array operation.
            '''
            return np.nonzero(age_cutoffs <= age)[0][-1]  # Index of the age bin to use

        cvu.set_seed(pars['rand_seed'])

        progs = pars['prognoses'] # Shorten the name
        inds = np.fromiter((find_cutoff(progs['age_cutoffs'], this_age) for this_age in self.age), dtype=cvd.default_int, count=len(self)) # Convert ages to indices
        self.symp_prob[:]   = progs['symp_probs'][inds] # Probability of developing symptoms
        self.severe_prob[:] = progs['severe_probs'][inds]*progs['comorbidities'][inds] # Severe disease probability is modified by comorbidities
        self.crit_prob[:]   = progs['crit_probs'][inds] # Probability of developing critical disease
        self.death_prob[:]  = progs['death_probs'][inds] # Probability of death
        self.rel_sus[:]     = progs['sus_ORs'][inds] # Default susceptibilities
        self.rel_trans[:]   = progs['trans_ORs'][inds]*cvu.sample(**self.pars['beta_dist'], size=len(inds)) # Default transmissibilities, with viral load drawn from a distribution

        return


    def update_states_pre(self, t):
        ''' Perform all state updates at the current timestep '''

        # Initialize
        self.t = t
        self.is_exp = self.true('exposed') # For storing the interim values since used in every subsequent calculation

        # Perform updates
        self.flows  = {key:0 for key in cvd.new_result_flows}
        self.flows['new_infectious']  += self.check_infectious() # For people who are exposed and not infectious, check if they begin being infectious
        self.flows['new_symptomatic'] += self.check_symptomatic()
        self.flows['new_severe']      += self.check_severe()
        self.flows['new_critical']    += self.check_critical()
        self.flows['new_deaths']      += self.check_death()
        self.flows['new_recoveries']  += self.check_recovery()

        return

    def update_states_post(self):
        ''' Perform post-timestep updates '''
        self.flows['new_diagnoses']   += self.check_diagnosed()
        self.flows['new_quarantined'] += self.check_quar()
        del self.is_exp  # Tidy up

        return


    def update_contacts(self):
        ''' Refresh dynamic contacts, e.g. community '''

        # Figure out if anything needs to be done -- e.g. {'h':False, 'c':True}
        dynam_keys = [lkey for lkey,is_dynam in self.pars['dynam_layer'].items() if is_dynam]

        # Loop over dynamic keys
        for lkey in dynam_keys:
            # Remove existing contacts
            self.contacts.pop(lkey)

            # Choose how many contacts to make
            pop_size   = len(self)
            n_contacts = self.pars['contacts'][lkey]
            n_new = int(n_contacts*pop_size/2) # Since these get looped over in both directions later

            # Create the contacts
            new_contacts = {} # Initialize
            new_contacts['p1']   = np.array(cvu.choose_r(max_n=pop_size, n=n_new), dtype=cvd.default_int) # Choose with replacement
            new_contacts['p2']   = np.array(cvu.choose_r(max_n=pop_size, n=n_new), dtype=cvd.default_int)
            new_contacts['beta'] = np.ones(n_new, dtype=cvd.default_float)

            # Add to contacts
            self.add_contacts(new_contacts, lkey=lkey)
            self.contacts[lkey].validate()

        return self.contacts


    #%% Methods for updating state

    def check_inds(self, current, date, filter_inds=None):
        ''' Return indices for which the current state is false and which meet the date criterion '''
        if filter_inds is None:
            not_current = cvu.false(current)
        else:
            not_current = cvu.ifalsei(current, filter_inds)
        has_date = cvu.idefinedi(date, not_current)
        inds     = cvu.itrue(self.t >= date[has_date], has_date)
        return inds


    def check_infectious(self):
        ''' Check if they become infectious '''
        inds = self.check_inds(self.infectious, self.date_infectious, filter_inds=self.is_exp)
        self.infectious[inds] = True
        return len(inds)


    def check_symptomatic(self):
        ''' Check for new progressions to symptomatic '''
        inds = self.check_inds(self.symptomatic, self.date_symptomatic, filter_inds=self.is_exp)
        self.symptomatic[inds] = True
        return len(inds)


    def check_severe(self):
        ''' Check for new progressions to severe '''
        inds = self.check_inds(self.severe, self.date_severe, filter_inds=self.is_exp)
        self.severe[inds] = True
        return len(inds)


    def check_critical(self):
        ''' Check for new progressions to critical '''
        inds = self.check_inds(self.critical, self.date_critical, filter_inds=self.is_exp)
        self.critical[inds] = True
        return len(inds)


    def check_recovery(self):
        ''' Check for recovery '''
        inds = self.check_inds(self.recovered, self.date_recovered, filter_inds=self.is_exp)
        self.exposed[inds]     = False
        self.infectious[inds]  = False
        self.symptomatic[inds] = False
        self.severe[inds]      = False
        self.critical[inds]    = False
        self.recovered[inds]   = True
        return len(inds)


    def check_death(self):
        ''' Check whether or not this person died on this timestep  '''
        inds = self.check_inds(self.dead, self.date_dead, filter_inds=self.is_exp)
        self.exposed[inds]     = False
        self.infectious[inds]  = False
        self.symptomatic[inds] = False
        self.severe[inds]      = False
        self.critical[inds]    = False
        self.recovered[inds]   = False
        self.dead[inds]        = True
        return len(inds)


    def check_diagnosed(self):
        '''
        Check for new diagnoses. Since most data are reported with diagnoses on
        the date of the test, this function reports counts not for the number of
        people who received a positive test result on a day, but rather, the number
        of people who were tested on that day who are schedule to be diagnosed in
        the future.
        '''

        # Handle people who tested today who will be diagnosed in future
        test_pos_inds = self.check_inds(self.diagnosed, self.date_pos_test, filter_inds=None) # Find people who will be diagnosed in future
        self.date_pos_test[test_pos_inds] = np.nan # Clear date of having will-be-positive test

        # Handle people who were actually diagnosed today
        diag_inds  = self.check_inds(self.diagnosed, self.date_diagnosed, filter_inds=None) # Find who was actually diagnosed on this timestep
        self.diagnosed[diag_inds]   = True # Set these people to be diagnosed
        self.quarantined[diag_inds] = False # If you are diagnosed, you are isolated, not in quarantine
        self.date_end_quarantine[diag_inds] = np.nan # Clear end quarantine time

        return len(test_pos_inds)


    def check_quar(self):
        ''' Check for who gets put into quarantine'''

        not_diagnosed_inds = self.false('diagnosed')
        all_inds = np.arange(len(self)) # Do dead people come out of quarantine?

        # Perform quarantine - on all who have a date_known_contact (Filter to those not already diagnosed?)
        not_quar_inds = self.check_inds(self.quarantined, self.date_known_contact, filter_inds=not_diagnosed_inds) # Check who is quarantined, not_diagnosed_inds?
        not_recovered = cvu.ifalsei(self.recovered, not_quar_inds)  # Pull out people who are not recovered
        quar_inds     = cvu.ifalsei(self.dead, not_recovered)       # ...or dead
        self.quarantine(quar_inds) # Put people in quarantine
        self.date_known_contact[quar_inds] = np.nan # Clear date

        # Check for the end of quarantine - on all who are quarantined
        end_inds = self.check_inds(~self.quarantined, self.date_end_quarantine, filter_inds=all_inds) # Note the double-negative here
        self.quarantined[end_inds] = False # Release from quarantine
        self.date_end_quarantine[end_inds] = np.nan # Clear end quarantine time

        n_quarantined = len(quar_inds)

        return n_quarantined


    #%% Methods to make events occur (infection and diagnosis)

    def make_susceptible(self, inds):
        '''
        Make person susceptible. This is used during dynamic resampling
        '''
        for key in self.meta.states:
            if key == 'susceptible':
                self[key][inds] = True
            else:
                self[key][inds] = False

        for key in self.meta.dates + self.meta.durs:
            self[key][inds] = np.nan

        return


    def infect(self, inds, hosp_max=None, icu_max=None, source=None, layer=None):
        '''
        Infect people and determine their eventual outcomes.
            * Every infected person can infect other people, regardless of whether they develop symptoms
            * Infected people that develop symptoms are disaggregated into mild vs. severe (=requires hospitalization) vs. critical (=requires ICU)
            * Every asymptomatic, mildly symptomatic, and severely symptomatic person recovers
            * Critical cases either recover or die

        Args:
            inds     (array): array of people to infect
            hosp_max (bool):  whether or not there is an acute bed available for this person
            icu_max  (bool):  whether or not there is an ICU bed available for this person
            source   (array): source indices of the people who transmitted this infection (None if an importation or seed infection)
            layer    (str):   contact layer this infection was transmitted on

        Returns:
            count (int): number of people infected
        '''

        # Remove duplicates
        unique = np.unique(inds, return_index=True)[1]
        inds = inds[unique]
        if source is not None:
            source = source[unique]

        # Keep only susceptibles
        keep = self.susceptible[inds] # Unique indices in inds and source that are also susceptible
        inds = inds[keep]
        if source is not None:
            source = source[keep]

        n_infections = len(inds)
        durpars      = self.pars['dur']

        # Set states
        self.susceptible[inds]   = False
        self.exposed[inds]       = True
        self.date_exposed[inds]  = self.t
        self.flows['new_infections'] += len(inds)

        # Record transmissions
        for i, target in enumerate(inds):
            self.infection_log.append(dict(source=source[i] if source is not None else None, target=target, date=self.t, layer=layer))

        # Calculate how long before this person can infect other people
        self.dur_exp2inf[inds] = cvu.sample(**durpars['exp2inf'], size=n_infections)
        self.date_infectious[inds] = self.dur_exp2inf[inds] + self.t

        # Use prognosis probabilities to determine what happens to them
        symp_probs = self.pars['rel_symp_prob']*self.symp_prob[inds] # Calculate their actual probability of being symptomatic
        is_symp = cvu.binomial_arr(symp_probs) # Determine if they develop symptoms
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
        sev_probs = self.pars['rel_severe_prob'] * self.severe_prob[symp_inds] # Probability of these people being severe
        is_sev = cvu.binomial_arr(sev_probs) # See if they're a severe or mild case
        sev_inds = symp_inds[is_sev]
        mild_inds = symp_inds[~is_sev] # Not severe

        # CASE 2.1: Mild symptoms, no hospitalization required and no probaility of death
        dur_mild2rec = cvu.sample(**durpars['mild2rec'], size=len(mild_inds))
        self.date_recovered[mild_inds] = self.date_symptomatic[mild_inds] + dur_mild2rec  # Date they recover
        self.dur_disease[mild_inds] = self.dur_exp2inf[mild_inds] + self.dur_inf2sym[mild_inds] + dur_mild2rec  # Store how long this person had COVID-19

        # CASE 2.2: Severe cases: hospitalization required, may become critical
        self.dur_sym2sev[sev_inds] = cvu.sample(**durpars['sym2sev'], size=len(sev_inds)) # Store how long this person took to develop severe symptoms
        self.date_severe[sev_inds] = self.date_symptomatic[sev_inds] + self.dur_sym2sev[sev_inds]  # Date symptoms become severe
        crit_probs = self.pars['rel_crit_prob'] * self.crit_prob[sev_inds] * (self.pars['no_hosp_factor'] if hosp_max else 1.)# Probability of these people becoming critical - higher if no beds available
        is_crit = cvu.binomial_arr(crit_probs)  # See if they're a critical case
        crit_inds = sev_inds[is_crit]
        non_crit_inds = sev_inds[~is_crit]

        # CASE 2.2.1 Not critical - they will recover
        dur_sev2rec = cvu.sample(**durpars['sev2rec'], size=len(non_crit_inds))
        self.date_recovered[non_crit_inds] = self.date_severe[non_crit_inds] + dur_sev2rec  # Date they recover
        self.dur_disease[non_crit_inds] = self.dur_exp2inf[non_crit_inds] + self.dur_inf2sym[non_crit_inds] + self.dur_sym2sev[non_crit_inds] + dur_sev2rec  # Store how long this person had COVID-19

        # CASE 2.2.2: Critical cases: ICU required, may die
        self.dur_sev2crit[crit_inds] = cvu.sample(**durpars['sev2crit'], size=len(crit_inds))
        self.date_critical[crit_inds] = self.date_severe[crit_inds] + self.dur_sev2crit[crit_inds]  # Date they become critical
        death_probs = self.pars['rel_death_prob'] * self.death_prob[crit_inds] * (self.pars['no_icu_factor'] if icu_max else 1.) # Probability they'll die
        is_dead = cvu.binomial_arr(death_probs)  # Death outcome
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

        inds = np.unique(inds)
        self.tested[inds] = True
        self.date_tested[inds] = self.t # Only keep the last time they tested

        is_infectious = cvu.itruei(self.infectious, inds)
        pos_test      = cvu.n_binomial(test_sensitivity, len(is_infectious))
        is_inf_pos    = is_infectious[pos_test]

        not_diagnosed = is_inf_pos[np.isnan(self.date_diagnosed[is_inf_pos])]
        not_lost      = cvu.n_binomial(1.0-loss_prob, len(not_diagnosed))
        final_inds    = not_diagnosed[not_lost]

        # Store the date the person will be diagnosed, as well as the date they took the test which will come back positive
        self.date_diagnosed[final_inds] = self.t + test_delay
        self.date_pos_test[final_inds] = self.t

        return


    def quarantine(self, inds):
        '''
        Quarantine selected people starting on the current day. If a person is already
        quarantined, this will extend their quarantine.
        Args:
            inds (array): indices of who to quarantine, specified by check_quar()
        '''
        self.quarantined[inds] = True
        self.date_quarantined[inds] = self.t
        self.date_end_quarantine[inds] = self.t + self.pars['quar_period']
        return


    def trace(self, inds, trace_probs, trace_time):
        '''
        Trace the contacts of the people provided
        Args:
            inds (array): indices of whose contacts to trace
            trace_probs (dict): probability of being able to trace people at each contact layer - should have the same keys as contacts
            trace_time (dict): days it'll take to trace people at each contact layer - should have the same keys as contacts
        '''

        # Extract the indices of the people who'll be contacted
        traceable_layers = {k:v for k,v in trace_probs.items() if v != 0.} # Only trace if there's a non-zero tracing probability
        for lkey,this_trace_prob in traceable_layers.items():
            if self.pars['beta_layer'][lkey]: # Skip if beta is 0 for this layer
                this_trace_time = trace_time[lkey]

                # Find all the contacts of these people
                nzinds = self.contacts[lkey]['beta'].nonzero()[0] # Find nonzero beta edges
                inds_list = []
                for k1,k2 in [['p1','p2'],['p2','p1']]: # Loop over the contact network in both directions
                    in_k1 = np.isin(self.contacts[lkey][k1], inds).nonzero()[0] # Get all the indices of the pairs that each person is in
                    nz_k1 = np.intersect1d(nzinds, in_k1) # Find the ones that are nonzero
                    inds_list.append(self.contacts[lkey][k2][nz_k1]) # Find their pairing partner
                edge_inds = np.unique(np.concatenate(inds_list)) # Find all edges

                # Check contacts
                contact_inds = cvu.binomial_filter(this_trace_prob, edge_inds) # Filter the indices according to the probability of being able to trace this layer
                if len(contact_inds):
                    self.known_contact[contact_inds] = True
                    self.date_known_contact[contact_inds]  = np.fmin(self.date_known_contact[contact_inds], self.t+this_trace_time)

        return


    #%% Analysis methods

    def plot(self, *args, **kwargs):
        '''
        Plot statistics of the population -- age distribution, numbers of contacts,
        and overall weight of contacts (number of contacts multiplied by beta per
        layer).

        Args:
            bins (arr): age bins to use (default, 0-100 in one-year bins)
            width (float): bar width
            font_size (float): size of font
            alpha (float): transparency of the plots
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            plot_args (dict): passed to pl.plot()
        '''
        fig = cvplt.plot_people(people=self, *args, **kwargs)
        return fig
