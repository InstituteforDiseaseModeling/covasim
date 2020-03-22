'''
This file contains all the code for the basic use of Covasim.

Version: 2020mar20
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import pylab as pl
import sciris as sc
import datetime as dt
import statsmodels.api as sm
import covasim.framework as cv
from . import parameters as cvpars


# Specify all externally visible functions this file defines
__all__ = ['to_plot', 'Person', 'Sim']

to_plot = sc.odict({
        'Total counts': sc.odict({
            'cum_exposed': 'Cumulative infections',
            'cum_deaths': 'Cumulative deaths',
            'cum_recoveries':'Cumulative recoveries',
            # 'cum_tested': 'Cumulative tested',
            # 'n_susceptible': 'Number susceptible',
            # 'n_infectious': 'Number of active infections',
            'cum_diagnosed': 'Cumulative diagnosed',
        }),
        'Daily counts': sc.odict({
            'infections': 'New infections',
            'deaths': 'New deaths',
            'recoveries': 'New recoveries',
            # 'tests': 'Number of tests',
            'diagnoses': 'New diagnoses',
        })
    })



#%% Define classes

class Person(cv.Person):
    '''
    Class for a single person.
    '''
    def __init__(self, age, sex, cfr, uid=None, id_len=4):
        if uid is None:
            uid = sc.uuid(length=id_len) # Unique identifier for this person
        self.uid  = str(uid)
        self.age  = float(age) # Age of the person (in years)
        self.sex  = int(sex) # Female (0) or male (1)
        self.cfr  = cfr # Case fatality rate

        # Define state
        self.alive          = True
        self.susceptible    = True
        self.exposed        = False
        self.infectious     = False
        self.symptomatic    = False
        self.diagnosed      = False
        self.recovered      = False
        self.dead           = False
        self.known_contact  = False # Keep track of whether each person is a contact of a known positive
        self.n_infected     = 0 # Keep track of how many people each person infects

        # Keep track of dates
        self.date_exposed     = None
        self.date_infectious  = None
        self.date_symptomatic = None
        self.date_diagnosed   = None
        self.date_recovered   = None
        self.date_died        = None
        return


class Sim(cv.Sim):
    '''
    The Sim class handles the running of the simulation: the number of children,
    number of time points, and the parameters of the simulation.
    '''

    def __init__(self, pars=None, datafile=None):
        if pars is None:
            pars = cvpars.make_pars()
        super().__init__(pars) # Initialize and set the parameters as attributes
        self.data = None # cvpars.load_data(datafile)
        self.stopped = None # If the simulation has stopped
        return


    def initialize(self):
        ''' Perform all initializations '''
        self.validate_pars()
        self.set_seed(self['seed'])
        self.init_results()
        self.init_people()
        return


    def validate_pars(self):
        ''' Some parameters can take multiple types; this makes them consistent '''

        # Handle start day
        start_day = self['start_day'] # Shorten
        if start_day in [None, 0]: # Use default start day
            start_day = dt.datetime(2020, 1, 1)
        if not isinstance(start_day, dt.datetime):
            errormsg = f"Start day must be a YYYY-MM-DD string or a datetime object, not {start_day}" # Not an error yet, but used twice, so pre-create it
            if sc.isstring(start_day):
                try:
                    start_day = dt.datetime.strptime(start_day, '%Y-%M-%d')
                except ValueError:
                    raise ValueError(errormsg) # Raise this
            else:
                raise TypeError(errormsg)
        self['start_day'] = start_day # Convert back

        # Replace tests with data, if available
        if self.data is not None:
            self['daily_tests'] = self.data['new_tests'] # Number of tests each day, from the data

        # Handle interventions
        for key in ['interv_days', 'interv_effs', 'daily_tests']:
            val = self[key]
            if val is None: val = [] # TODO: have skipnone be an option in promotetoarray()
            self[key] = sc.promotetoarray(self[key])

        return


    def init_results(self):
        ''' Initialize results '''

        def init_res(*args, **kwargs):
            ''' Initialize a single result object '''
            output = cv.Result(*args, **kwargs, npts=self.npts)
            return output

        # Create the main results structure
        self.results = {}
        self.results['n_susceptible']  = init_res('Number susceptible')
        self.results['n_exposed']      = init_res('Number exposed')
        self.results['n_infectious']   = init_res('Number infectious')
        self.results['n_symptomatic']  = init_res('Number symptomatic')
        self.results['n_recovered']    = init_res('Number recovered')
        self.results['infections']     = init_res('Number of new infections')
        self.results['tests']          = init_res('Number of tests')
        self.results['diagnoses']      = init_res('Number of new diagnoses')
        self.results['recoveries']     = init_res('Number of new recoveries')
        self.results['deaths']         = init_res('Number of new deaths')
        self.results['cum_exposed']    = init_res('Cumulative number exposed')
        self.results['cum_tested']     = init_res('Cumulative number of tests')
        self.results['cum_diagnosed']  = init_res('Cumulative number diagnosed')
        self.results['cum_deaths']     = init_res('Cumulative number of deaths')
        self.results['cum_recoveries'] = init_res('Cumulative number recovered')
        self.results['doubling_time']  = init_res('Doubling time', scale=False)
        self.results['r_eff']          = init_res('Effective reproductive number', scale=False)

        self.reskeys = list(self.results.keys()) # Save the names of the main result keys

        # Populate the rest of the results
        self.results['t'] = self.tvec
        self.results['date'] = [self['start_day'] + dt.timedelta(days=int(t)) for t in self.tvec]
        self.transtree = {} # For storing the transmission tree
        self.results_ready = False

        # Create calculated values structure
        self.calculated = {}
        self.calculated['eff_beta'] = self['asym_prop']*self['asym_factor']*self['beta'] + (1-self['asym_prop'])*self['beta']  # Using asymptomatic proportion
        self.calculated['r_0']      = self['contacts']*self['dur']*self.calculated['eff_beta']
        return


    def init_people(self, verbose=None, id_len=6):
        ''' Create the people '''
        if verbose is None:
            verbose = self['verbose']

        if verbose>=2:
            print(f'Creating {self["n"]} people...')

        # Create the people -- just placeholders if we're using actual data
        people = {} # Dictionary for storing the people -- use plain dict since faster than odict
        n_people = int(self['n'])
        uids = sc.uuid(which='ascii', n=n_people, length=id_len)
        for p in range(n_people): # Loop over each person
            uid = uids[p]
            if self['usepopdata']:
                age,sex,cfr = -1, -1, -1 # These get overwritten later
            else:
                age,sex,cfr = cvpars.get_age_sex(cfr_by_age=self['cfr_by_age'], default_cfr=self['default_cfr'], use_data=False)

            person = Person(age=age, sex=sex, cfr=cfr, uid=uid) # Create the person
            people[uid] = person # Save them to the dictionary

        # Store UIDs and people
        self.uids = uids
        self.people = people

        # Make the contact matrix
        if not self['usepopdata']:
            if verbose>=2:
                print(f'Creating contact matrix without data...')
            for p in range(int(self['n'])):
                person = self.get_person(p)
                person.n_contacts = cv.pt(self['contacts']) # Draw the number of Poisson contacts for this person
                person.contact_inds = cv.choose_people(max_ind=len(self.people), n=person.n_contacts) # Choose people at random, assigning to household
        else:
            if verbose>=2:
                print(f'Creating contact matrix with data...')
            import synthpops as sp

            self.contact_keys = self['contacts_pop'].keys()
            popdict = sp.make_popdict(uids=self.uids)
            popdict = sp.make_contacts(popdict, self['contacts'], use_social_layers=True)
            popdict = sc.odict(popdict)
            for p,uid,entry in popdict.enumitems():
                person = self.get_person(p)
                person.age = entry['age']
                person.sex = entry['sex']
                person.cfr = cvpars.get_cfr(person.age, default_cfr=self['default_cfr'], cfr_by_age=self['cfr_by_age'])
                person.contact_inds = entry['contacts']

        if verbose >= 1:
            print(f'Created {self["n"]} people, average age {sum([person.age for person in self.people.values()])/self["n"]}')

        # Create the seed infections
        for i in range(int(self['n_infected'])):
            person = self.get_person(i)
            self.infect_person(source_person=None, target_person=person, t=0)

        return


    def summary_stats(self, verbose=None):
        ''' Compute the summary statistics to display at the end of a run '''

        if verbose is None:
            verbose = self['verbose']

        summary = {}
        for key in self.reskeys:
            summary[key] = self.results[key][-1]

        if verbose:
            print(f"""Summary:
     {summary['n_susceptible']:5.0f} susceptible
     {summary['n_infectious']:5.0f} infectious
     {summary['n_symptomatic']:5.0f} symptomatic
     {summary['cum_exposed']:5.0f} total exposed
     {summary['cum_diagnosed']:5.0f} total diagnosed
     {summary['cum_deaths']:5.0f} total deaths
     {summary['cum_recoveries']:5.0f} total recovered
               """)

        return summary


    def infect_person(self, source_person, target_person, t, infectious=False):
        '''
        Infect target_person. source_person is used only for constructing the
        transmission tree.
        '''
        target_person.susceptible = False
        target_person.exposed = True
        target_person.date_exposed = t

        serial_pars = dict(dist='normal_int', par1=self['serial'],    par2=self['serial_std'])
        incub_pars  = dict(dist='normal_int', par1=self['incub'],     par2=self['incub_std'])
        dur_pars    = dict(dist='normal_int', par1=self['dur'],       par2=self['dur_std'])
        death_pars  = dict(dist='normal_int', par1=self['timetodie'], par2=self['timetodie_std'])

        # Calculate how long before they can infect other people
        serial_dist = cv.sample(**serial_pars)
        target_person.date_infectious = t + serial_dist

        # Program them to either die or recover in the future
        if cv.bt(target_person.cfr):
            # They die
            death_dist = cv.sample(**death_pars)
            target_person.date_died = t + death_dist
        else:
            # They don't die; determine whether they develop symptoms
            # TODO, consider refactoring this with a "symptom_severity" parameter that could help determine likelihood of hospitalization
            if not cv.bt(self['asym_prop']): # They develop symptoms
                incub_dist = cv.sample(**incub_pars) # Caclulate how long til they develop symptoms
                target_person.date_symptomatic = t + incub_dist

            dur_dist = cv.sample(**dur_pars)
            target_person.date_recovered = target_person.date_infectious + dur_dist

        if source_person:
            self.transtree[target_person.uid] = {'from':source_person.uid, 'date':t}

        return target_person


    def run(self, verbose=None, calc_likelihood=False, do_plot=False, **kwargs):
        ''' Run the simulation '''

        T = sc.tic()

        # Reset settings and results
        if verbose is None:
            verbose = self['verbose']
        self.initialize() # Create people, results, etc.

        # Main simulation loop
        self.stopped = False # We've just been asked to run, so ensure we're unstopped
        for t in range(self.npts):

            # Check timing and stopping function
            elapsed = sc.toc(T, output=True)
            if elapsed > self['timelimit']:
                print(f"Time limit ({self['timelimit']} s) exceeded; stopping...")
                self.stopped = {'why':'timelimit', 'message':'Time limit exceeded at step {t}', 't':t}

            if self['stop_func']:
                self.stopped = self['stop_func'](self) # Feed in the current simulation object

            # If this gets set, stop running -- e.g. if the time limit is exceeded
            if self.stopped:
                break

            # Print progress
            if verbose>=1:
                string = f'  Running day {t:0.0f} of {self.pars["n_days"]} ({elapsed:0.2f} s elapsed)...'
                if verbose>=2:
                    sc.heading(string)
                else:
                    print(string)

            test_probs = {} # Store the probability of each person getting tested

            # Update each person
            for person in self.people.values():

                # Initialise testing -- assign equal testing probabilities initially, these will get adjusted later
                test_probs[person.uid] = 1.0

                # Count susceptibles
                if person.susceptible:
                    self.results['n_susceptible'][t] += 1
                    continue # Don't bother with the rest of the loop

                # If exposed, check if the person becomes infectious or develops symptoms
                if person.exposed:
                    self.results['n_exposed'][t] += 1
                    if not person.infectious and t >= person.date_infectious: # It's the day they become infectious
                        person.infectious = True
                        if verbose >= 2:
                            print(f'      Person {person.uid} became infectious!')
                    if not person.symptomatic and person.date_symptomatic is not None and t >= person.date_symptomatic:  # It's the day they develop symptoms
                        person.symptomatic = True
                        if verbose >= 2:
                            print(f'      Person {person.uid} developed symptoms!')

                # If infectious, check if anyone gets infected
                if person.infectious:

                    # Check for death
                    if person.date_died and t >= person.date_died:
                        person.exposed = False
                        person.infectious = False
                        person.symptomatic = False
                        person.recovered = False
                        person.died = True
                        self.results['deaths'][t] += 1

                    # Check for recovery
                    if person.date_recovered and t >= person.date_recovered: # It's the day they recover
                        person.exposed = False
                        person.infectious = False
                        person.symptomatic = False
                        person.recovered = True
                        self.results['recoveries'][t] += 1

                    # Calculate onward transmission
                    else:
                        self.results['n_infectious'][t] += 1 # Count this person as infectious
                        if not self['usepopdata']: # TODO: refactor!

                            for contact_ind in person.contact_inds:

                                target_person = self.get_person(contact_ind)  # Stored by integer

                                # This person was diagnosed last time step: time to flag their contacts
                                if person.date_diagnosed is not None and person.date_diagnosed == t-1:
                                    target_person.known_contact = True

                                # Calculate transmission risk based on whether they're asymptomatic/diagnosed/have been isolated
                                thisbeta = self['beta'] * \
                                           (self['asym_factor'] if person.symptomatic else 1.) * \
                                           (self['diag_factor'] if person.diagnosed else 1.) * \
                                           (self['cont_factor'] if person.known_contact else 1.)
                                transmission = cv.bt(thisbeta) # Check whether virus is transmitted

                                if transmission:
                                    if target_person.susceptible: # Skip people who are not susceptible
                                        self.results['infections'][t] += 1
                                        self.infect_person(source_person=person, target_person=target_person, t=t)
                                        person.n_infected += 1
                                        if verbose>=2:
                                            print(f'        Person {person.uid} infected person {target_person.uid}!')

                        else:

                            for ckey in self.contact_keys:

                                # Calculate transmission risk based on whether they're asymptomatic/diagnosed
                                for contact_ind in person.contact_inds[ckey]:
                                    thisbeta = self['beta'] * self['beta_pop'][ckey]  * \
                                               (self['asym_factor'] if person.symptomatic else 1.) * \
                                               (self['diag_factor'] if person.diagnosed else 1.) * \
                                               (self['cont_factor'] if person.known_contact else 1.)
                                    transmission = cv.bt(thisbeta) # Check whether virus is transmitted

                                    if transmission:
                                        target_person = self.people[contact_ind] # Stored by UID
                                        if target_person.susceptible: # Skip people who are not susceptible
                                            self.results['infections'][t] += 1
                                            self.infect_person(source_person=person, target_person=target_person, t=t)
                                            person.n_infected += 1
                                            if verbose>=2:
                                                print(f'        Person {person.uid} infected person {target_person.uid} via {ckey}!')


                # Count people who developed symptoms
                if person.symptomatic:
                    self.results['n_symptomatic'][t] += 1

                # Count people who recovered
                if person.recovered:
                    self.results['n_recovered'][t] += 1

                # Adjust testing probability based on what's happened to the person
                # NB, these need to be separate if statements, because a person can be both diagnosed and infectious/symptomatic
                if person.symptomatic:
                    test_probs[person.uid] *= self['sympt_test'] # They're symptomatic
                if person.known_contact:
                    test_probs[person.uid] *= self['trace_test']  # They've had contact with a known positive
                if person.diagnosed:
                    test_probs[person.uid] = 0.0

            # Implement testing -- this is outside of the loop over people, but inside the loop over time
            if t<len(self['daily_tests']): # Don't know how long the data is, ensure we don't go past the end
                n_tests = self['daily_tests'][t] # Number of tests for this day
                if n_tests and not pl.isnan(n_tests): # There are tests this day
                    self.results['tests'][t] = n_tests # Store the number of tests
                    test_probs_arr = pl.array(list(test_probs.values()))
                    test_probs_arr /= test_probs_arr.sum()
                    test_inds = cv.choose_people_weighted(probs=test_probs_arr, n=n_tests)

                    for test_ind in test_inds:
                        tested_person = self.get_person(test_ind)
                        if tested_person.infectious and cv.bt(self['sensitivity']): # Person was tested and is true-positive
                            self.results['diagnoses'][t] += 1
                            tested_person.diagnosed = True
                            tested_person.date_diagnosed = t
                            if verbose>=2:
                                print(f'          Person {tested_person.uid} was diagnosed at timestep {t}!')

            # Implement quarantine
            if t in self['interv_days']:
                ind = sc.findinds(self['interv_days'], t)[0]
                eff = self['interv_effs'][ind]
                if verbose>=1:
                    print(f'Implementing intervention/change of {eff} on day {t}...')
                self['beta'] *= eff # TODO: pop-specific

            # Calculate doubling time
            cum_infections = pl.cumsum(self.results['infections'][:t+1])
            nonzero = np.nonzero(cum_infections)[0] # Skip days with zero infections for the log2 below
            if len(nonzero) >= 2: # Need at least 2 points
                exog  = sm.add_constant(self.tvec[nonzero])
                endog = np.log2(cum_infections[nonzero])
                model = sm.OLS(endog, exog)
                doubling_rate = model.fit().params[1]
                if doubling_rate != 0: # If it's zero, skip
                    doubling_time = 1.0 / doubling_rate
                    self.results['doubling_time'][t] = doubling_time

            # Effective reproductive number based on number still susceptible
            self.results['r_eff'][t] = self.calculated['r_0']*self.results['n_susceptible'][t]/self['n']

        # Compute cumulative results
        self.results['cum_exposed'].values    = pl.cumsum(self.results['infections'].values) + self['n_infected'] # Include initially infected people
        self.results['cum_tested'].values     = pl.cumsum(self.results['tests'].values)
        self.results['cum_diagnosed'].values  = pl.cumsum(self.results['diagnoses'].values)
        self.results['cum_deaths'].values     = pl.cumsum(self.results['deaths'].values)
        self.results['cum_recoveries'].values = pl.cumsum(self.results['recoveries'].values)


        # Scale the results
        for reskey in self.reskeys:
            if self.results[reskey].scale:
                self.results[reskey].values *= self['scale']

        # Compute likelihood
        if calc_likelihood:
            self.likelihood()

        # Tidy up
        self.results_ready = True
        if verbose>=1:
            print(f'\nRun finished after {elapsed:0.1f} s.\n')
            self.results['summary'] = self.summary_stats()

        if do_plot:
            self.plot(**kwargs)

        # Convert to an odict to allow e.g. sim.people[25] later
        self.people = sc.odict(self.people)

        return self.results


    def likelihood(self, verbose=None):
        '''
        Compute the log-likelihood of the current simulation based on the number
        of new diagnoses.
        '''
        if verbose is None:
            verbose = self['verbose']

        if not self.results_ready:
            self.run(calc_likelihood=False, verbose=verbose) # To avoid an infinite loop

        loglike = 0
        if self.data is not None and len(self.data):
            for d,datum in enumerate(self.data['new_positives']):
                if not pl.isnan(datum): # Skip days when no tests were performed
                    estimate = self.results['diagnoses'][d]
                    p = cv.poisson_test(datum, estimate)
                    logp = pl.log(p)
                    loglike += logp
                    if verbose>=2:
                        print(f'  {self.data["date"][d]}, data={datum:3.0f}, model={estimate:3.0f}, log(p)={logp:10.4f}, loglike={loglike:10.4f}')

        self.results['likelihood'] = loglike

        if verbose>=1:
            print(f'Likelihood: {loglike}')

        return loglike



    def plot(self, do_save=None, fig_path=None, fig_args=None, plot_args=None,
             scatter_args=None, axis_args=None, as_dates=True, interval=None, dateformat=None,
             font_size=18, font_family=None, use_grid=True, do_show=True, verbose=None):
        '''
        Plot the results -- can supply arguments for both the figure and the plots.

        Args:
            do_save (bool or str): Whether or not to save the figure. If a string, save to that filename.
            fig_path (str): Path to save the figure
            fig_args (dict): Dictionary of kwargs to be passed to pl.figure()
            plot_args (dict): Dictionary of kwargs to be passed to pl.plot()
            scatter_args (dict): Dictionary of kwargs to be passed to pl.scatter()
            axis_args (dict): Dictionary of kwargs to be passed to pl.subplots_adjust()
            as_dates (bool): Whether to plot the x-axis as dates or time points
            interval (int): Interval between tick marks
            dateformat (str): Date string format, e.g. '%B %d'
            font_size (int): Size of the font
            font_family (str): Font face
            use_grid (bool): Whether or not to plot gridlines
            do_show (bool): Whether or not to show the figure
            verbose (bool): Display a bit of extra information

        Returns:
            fig: Figure handle
        '''

        if verbose is None:
            verbose = self['verbose']
        if verbose:
            print('Plotting...')

        if fig_args     is None: fig_args     = {'figsize':(16,12)}
        if plot_args    is None: plot_args    = {'lw':3, 'alpha':0.7}
        if scatter_args is None: scatter_args = {'s':150, 'marker':'s'}
        if axis_args    is None: axis_args    = {'left':0.1, 'bottom':0.05, 'right':0.9, 'top':0.97, 'wspace':0.2, 'hspace':0.25}

        fig = pl.figure(**fig_args)
        pl.subplots_adjust(**axis_args)
        pl.rcParams['font.size'] = font_size
        if font_family:
            pl.rcParams['font.family'] = font_family

        res = self.results # Shorten since heavily used

        # Plot everything

        colors = sc.gridcolors(max([len(tp) for tp in to_plot.values()]))

        # Define the data mapping. Must be here since uses functions
        if self.data is not None and len(self.data):
            data_mapping = {
                'cum_exposed': pl.cumsum(self.data['new_infections']),
                'cum_diagnosed':  pl.cumsum(self.data['new_positives']),
                'cum_tested':     pl.cumsum(self.data['new_tests']),
                'infections':     self.data['new_infections'],
                'tests':          self.data['new_tests'],
                'diagnoses':      self.data['new_positives'],
                }
        else:
            data_mapping = {}

        for p,title,keylabels in to_plot.enumitems():
            ax = pl.subplot(2,1,p+1)
            for i,key,label in keylabels.enumitems():
                this_color = colors[i]
                y = res[key].values
                pl.plot(res['t'], y, label=label, **plot_args, c=this_color)
                if key in data_mapping:
                    pl.scatter(self.data['day'], data_mapping[key], c=[this_color], **scatter_args)
            if self.data is not None and len(self.data):
                pl.scatter(pl.nan, pl.nan, c=[(0,0,0)], label='Data', **scatter_args)
            pl.grid(use_grid)
            cv.fixaxis(self)
            sc.commaticks()
            pl.title(title)

            # Optionally reset tick marks (useful for e.g. plotting weeks/months)
            if interval:
                xmin,xmax = ax.get_xlim()
                ax.set_xticks(pl.arange(xmin, xmax+1, interval))

            # Set xticks as dates
            if as_dates:
                xticks = ax.get_xticks()
                xticklabels = self.inds2dates(xticks, dateformat=dateformat)
                ax.set_xticklabels(xticklabels)

        # Ensure the figure actually renders or saves
        if do_save:
            if isinstance(do_save, str) and fig_path is None:
                fig_path = do_save # It's a string, assume it's a filename
            else:
                fig_path = 'covasim.png' # Just give it a default name
            pl.savefig(fig_path)

        if do_show:
            pl.show()
        else:
            pl.close(fig)

        return fig


    def plot_people(self):
        ''' Use imshow() to show all individuals as rows, with time as columns, one pixel per timestep per person '''
        raise NotImplementedError
