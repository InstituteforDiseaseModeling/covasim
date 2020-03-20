'''
This file contains all the code for a single run of Covid-ABM.

Based heavily on LEMOD-FP (https://github.com/amath-idm/lemod_fp).

Version: 2020mar13
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import pylab as pl
import sciris as sc
import datetime as dt
import statsmodels.api as sm
import covasim.cova_base as cova
from . import parameters as cova_pars


# Specify all externally visible functions this file defines
__all__ = ['to_plot', 'Person', 'Sim']

to_plot = sc.odict({
        'Total counts': sc.odict({
            'cum_exposed': 'Cumulative infections',
            'cum_deaths': 'Cumulative deaths',
            'cum_recoveries':'Cumulative recoveries',
            'cum_tested': 'Cumulative tested',
            # 'n_susceptible': 'Number susceptible',
            # 'n_infectious': 'Number of active infections',
            'cum_diagnosed': 'Cumulative diagnosed',
        }),
        'Daily counts': sc.odict({
            'infections': 'New infections',
            'deaths': 'New deaths',
            'recoveries': 'New recoveries',
            'tests': 'Number of tests',
            'diagnoses': 'New diagnoses',
        })
    })



#%% Define classes

class Person(cova.Person):
    '''
    Class for a single person.
    '''
    def __init__(self, pars, age=0, sex=0, cfr=0, uid=None, id_len=4):
        super().__init__(pars) # Set parameters
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


class Result(sc.prettyobj): #TODO: make this a general class?
    '''
    Stores a single result
    '''
    def __init__(self, name=None, scale=True, ispercentage=False, values=None):
        self.name = name  # Name of this result
        self.ispercentage = ispercentage  # Whether or not the result is a percentage
        self.scale = scale  # Whether or not to scale the result by the scale factor
        self.values = values
        return


class Sim(cova.Sim):
    '''
    The Sim class handles the running of the simulation: the number of children,
    number of time points, and the parameters of the simulation.
    '''

    def __init__(self, pars=None, datafile=None):
        if pars is None:
            pars = cova_pars.make_pars()
        super().__init__(pars) # Initialize and set the parameters as attributes
        self.data = None # cova_pars.load_data(datafile)
        self.set_seed(self['seed'])
        self.init_results()
        self.init_people()
        self.interventions = {}
        return


    def init_results(self):
        ''' Initialize results '''

        self.results = {}
        self.results['n_susceptible']       = Result('Number susceptible')
        self.results['n_exposed']           = Result('Number exposed')
        self.results['n_infectious']        = Result('Number infectious')
        self.results['n_symptomatic']       = Result('Number symptomatic')
        self.results['n_recovered']         = Result('Number recovered')
        self.results['infections']          = Result('Number of new infections')
        self.results['tests']               = Result('Number of tests')
        self.results['diagnoses']           = Result('Number of new diagnoses')
        self.results['recoveries']          = Result('Number of new recoveries')
        self.results['deaths']              = Result('Number of new deaths')
        self.results['cum_exposed']         = Result('Cumulative number exposed')
        self.results['cum_tested']          = Result('Cumulative number of tests')
        self.results['cum_diagnosed']       = Result('Cumulative number diagnosed')
        self.results['cum_deaths']          = Result('Cumulative number of deaths')
        self.results['cum_recoveries']      = Result('Cumulative number recovered')
        self.results['doubling_time']       = Result('Doubling time', scale=False)

        self.reskeys = [k for k in self.results.keys() if isinstance(self.results[k],Result)] # Save the names of the main result keys

        for key in self.reskeys:
            self.results[key].values = np.zeros(int(self.npts))

        self.results['t'] = np.arange(int(self.npts))
        self.results['transtree'] = {} # For storing the transmission tree
        self.results['ready'] = False
        return


    def init_people(self, verbose=None, id_len=4):
        ''' Create the people '''
        if verbose is None:
            verbose = self['verbose']

        if verbose>=2:
            print(f'Creating {self["n"]} people...')

        self.people = {} # Dictionary for storing the people -- use plain dict since faster

        for p in range(int(self['n'])): # Loop over each person
            if self['usepopdata']:
                age,sex,cfr = -1, -1, 0 # These get overwritten later
            else:
                age,sex,cfr = cova_pars.get_age_sex(cfr_by_age=self['cfr_by_age'], use_data=False)
            uid = None
            while not uid or uid in self.people.keys():
                uid = sc.uuid(length=id_len)

            person = Person(self.pars, age=age, sex=sex, cfr=cfr, uid=uid) # Create the person
            self.people[person.uid] = person # Save them to the dictionary

        if verbose >= 1:
            print(f'Created {self["n"]} people, average age {sum([person.age for person in self.people.values()])/self["n"]}')

        # Store all the UIDs as a list
        self.uids = list(self.people.keys())

        # Create the seed infections
        for i in range(int(self['n_infected'])):
            self.results['infections'].values[0] += 1
            person = self.get_person(i)
            person.susceptible = False
            person.exposed = True
            person.infectious = True
            person.symptomatic = True # Assume they have symptoms
            person.date_exposed = 0
            person.date_infectious = 0
            person.date_symptomatic = 0

        # Make the contact matrix
        if not self['usepopdata']:
            if verbose>=2:
                print(f'Creating contact matrix without data...')
            for p in range(int(self['n'])):
                person = self.get_person(p)
                person.n_contacts = cova.pt(person['contacts']) # Draw the number of Poisson contacts for this person
                person.contact_inds = cova.choose_people(max_ind=len(self.people), n=person.n_contacts) # Choose people at random, assigning to household
        else:
            self.contact_keys = self['contacts_pop'].keys()
            if verbose>=2:
                print(f'Creating contact matrix with data...')
            import synthpops as sp
            popdict = sp.make_popdict(uids=self.uids)
            popdict = sp.make_contacts(popdict, self['contacts'], use_social_layers=True)
            popdict = sc.odict(popdict)
            for p,uid,entry in popdict.enumitems():
                person = self.get_person(p)
                person.age = entry['age']
                person.sex = entry['sex']
                person.contact_inds = entry['contacts']

        return


    def summary_stats(self, verbose=None):
        ''' Compute the summary statistics to display at the end of a run '''

        if verbose is None:
            verbose = self['verbose']

        summary = {}
        for key in self.reskeys:
            summary[key] = self.results[key].values[-1]

        if verbose:
            print(f"""Summary:
     {summary['n_susceptible']:5.0f} susceptible
     {summary['n_infectious']:5.0f} infectious
     {summary['n_symptomatic']:5.0f} symptomatic
     {summary['cum_exposed']:5.0f} exposed
     {summary['cum_diagnosed']:5.0f} diagnosed
     {summary['cum_deaths']:5.0f} deaths
     {summary['cum_recoveries']:5.0f} recovered
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

        serial_pars = dict(dist='normal_int', par1=target_person.pars['serial'],    par2=target_person.pars['serial_std'])
        incub_pars = dict(dist='normal_int',  par1=target_person.pars['incub'],     par2=target_person.pars['incub_std'])
        dur_pars   = dict(dist='normal_int',  par1=target_person.pars['dur'],       par2=target_person.pars['dur_std'])
        death_pars = dict(dist='normal_int',  par1=target_person.pars['timetodie'], par2=target_person.pars['timetodie_std'])

        # Calculate how long before they can infect other people
        serial_dist = cova.sample(**serial_pars)
        target_person.date_infectious = t + serial_dist

        # Program them to either die or recover in the future
        if cova.bt(target_person.cfr):
            # They die
            death_dist = cova.sample(**death_pars)
            target_person.date_died = t + death_dist
        else:
            # They don't die; determine whether they develop symptoms
            # TODO, consider refactoring this with a "symptom_severity" parameter that could help determine likelihood of hospitalization
            if not cova.bt(target_person.pars['asymptomatic']): # They develop symptoms
                incub_dist = cova.sample(**incub_pars) # Caclulate how long til they develop symptoms
                target_person.date_symptomatic = t + incub_dist

            dur_dist = cova.sample(**dur_pars)
            target_person.date_recovered = target_person.date_infectious + dur_dist

        self.results['transtree'][target_person.uid] = {'from':source_person.uid, 'date':t}

        return target_person


    def run(self, verbose=None, calc_likelihood=False, do_plot=False, **kwargs):
        ''' Run the simulation '''

        T = sc.tic()

        # Reset settings and results
        if verbose is None:
            verbose = self['verbose']
        self.init_results()
        self.init_people() # Actually create the people
        if self.data is not None and len(self.data): # TODO: refactor to single conditional
            daily_tests = self.data['new_tests'] # Number of tests each day, from the data
        else:
            daily_tests = self['daily_tests']

        # Main simulation loop
        for t in range(self.npts):

            # Print progress
            if verbose>=1:
                string = f'  Running day {t:0.0f} of {self.pars["n_days"]}...'
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
                    self.results['n_susceptible'].values[t] += 1
                    continue # Don't bother with the rest of the loop

                # If exposed, check if the person becomes infectious or develops symptoms
                if person.exposed:
                    self.results['n_exposed'].values[t] += 1
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
                        self.results['deaths'].values[t] += 1

                    # Check for recovery
                    if person.date_recovered and t >= person.date_recovered: # It's the day they recover
                        person.exposed = False
                        person.infectious = False
                        person.symptomatic = False
                        person.recovered = True
                        self.results['recoveries'].values[t] += 1

                    # Calculate onward transmission
                    else:
                        self.results['n_infectious'].values[t] += 1 # Count this person as infectious
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
                                transmission = cova.bt(thisbeta) # Check whether virus is transmitted

                                if transmission:
                                    if target_person.susceptible: # Skip people who are not susceptible
                                        self.results['infections'].values[t] += 1
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
                                    transmission = cova.bt(thisbeta) # Check whether virus is transmitted

                                    if transmission:
                                        target_person = self.people[contact_ind] # Stored by UID
                                        if target_person.susceptible: # Skip people who are not susceptible
                                            self.results['infections'].values[t] += 1
                                            self.infect_person(source_person=person, target_person=target_person, t=t)
                                            person.n_infected += 1
                                            if verbose>=2:
                                                print(f'        Person {person.uid} infected person {target_person.uid} via {ckey}!')


                # Count people who developed symptoms
                if person.symptomatic:
                    self.results['n_symptomatic'].values[t] += 1

                # Count people who recovered
                if person.recovered:
                    self.results['n_recovered'].values[t] += 1

                # Adjust testing probability based on what's happened to the person
                # NB, these need to be separate if statements, because a person can be both diagnosed and infectious/symptomatic
                if person.symptomatic:
                    test_probs[person.uid] *= self['sympt_test'] # They're symptomatic
                if person.known_contact:
                    test_probs[person.uid] *= self['trace_test']  # They've had contact with a known positive
                if person.diagnosed:
                    test_probs[person.uid] = 0.0

            # Implement testing -- this is outside of the loop over people, but inside the loop over time
            if t<len(daily_tests): # Don't know how long the data is, ensure we don't go past the end
                n_tests = daily_tests.iloc[t] # Number of tests for this day
                if n_tests and not pl.isnan(n_tests): # There are tests this day
                    self.results['tests'].values[t] = n_tests # Store the number of tests
                    test_probs_arr = pl.array(list(test_probs.values()))
                    test_probs_arr /= test_probs_arr.sum()
                    test_inds = cova.choose_people_weighted(probs=test_probs_arr, n=n_tests)

                    for test_ind in test_inds:
                        tested_person = self.get_person(test_ind)
                        if tested_person.infectious and cova.bt(self['sensitivity']): # Person was tested and is true-positive
                            self.results['diagnoses'].values[t] += 1
                            tested_person.diagnosed = True
                            tested_person.date_diagnosed = t
                            if verbose>=2:
                                        print(f'          Person {tested_person.uid} was diagnosed at timestep {t}!')

            # Implement quarantine
            if t in self['interv_days']:
                if verbose>=1:
                    print(f'Implementing intervention/change on day {t}...')
                ind = sc.findinds(self['interv_days'], t)[0]
                self['beta'] *= self['interv_effs'][ind] # TODO: pop-specific

            # Doubling time
            if t>=1:
                exog  = sm.add_constant(np.arange(t+1))
                endog = np.log2(pl.cumsum(self.results['infections'].values[:t+1]))
                model = sm.OLS(endog, exog)
                doubling_time = 1 / model.fit().params[1]
                self.results['doubling_time'].values[t] = doubling_time

        # Compute cumulative results
        self.results['cum_exposed'].values    = pl.cumsum(self.results['infections'].values)
        self.results['cum_tested'].values     = pl.cumsum(self.results['tests'].values)
        self.results['cum_diagnosed'].values  = pl.cumsum(self.results['diagnoses'].values)
        self.results['cum_deaths'].values     = pl.cumsum(self.results['deaths'].values)
        self.results['cum_recoveries'].values = pl.cumsum(self.results['recoveries'].values)

        # Scale the results
        for reskey in self.reskeys:
            try:
                if self.results[reskey].scale:
                    self.results[reskey].values *= self['scale']
            except:
                import traceback;
                traceback.print_exc();
                import pdb;
                pdb.set_trace()

        # Compute likelihood
        if calc_likelihood:
            self.likelihood()

        # Tidy up
        self.results['ready'] = True
        elapsed = sc.toc(T, output=True)
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

        if not self.results['ready']:
            self.run(calc_likelihood=False, verbose=verbose) # To avoid an infinite loop

        loglike = 0
        if self.data is not None and len(self.data):
            for d,datum in enumerate(self.data['new_positives']):
                if not pl.isnan(datum): # Skip days when no tests were performed
                    estimate = self.results['diagnoses'][d]
                    p = cova.poisson_test(datum, estimate)
                    logp = pl.log(p)
                    loglike += logp
                    if verbose>=2:
                        print(f'  {self.data["date"][d]}, data={datum:3.0f}, model={estimate:3.0f}, log(p)={logp:10.4f}, loglike={loglike:10.4f}')

        self.results['likelihood'] = loglike

        if verbose>=1:
            print(f'Likelihood: {loglike}')

        return loglike



    def plot(self, do_save=None, fig_path=None, fig_args=None, plot_args=None, scatter_args=None, axis_args=None, as_days=True, font_size=18, use_grid=True, verbose=None):
        '''
        Plot the results -- can supply arguments for both the figure and the plots.

        Args:
            do_save (bool or str): Whether or not to save the figure. If a string, save to that filename.
            fig_args (dict): Dictionary of kwargs to be passed to pl.figure()
            plot_args (dict): Dictionary of kwargs to be passed to pl.plot()
            as_days (bool) Whether to plot the x-axis as days or time points

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
        pl.rcParams['font.family'] = 'Proxima Nova'

        res = self.results # Shorten since heavily used

        # Plot everything

        colors = sc.gridcolors(max([len(tp) for tp in to_plot.values()]))

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
            pl.subplot(2,1,p+1)
            for i,key,label in keylabels.enumitems():
                this_color = colors[i]
                y = res[key]
                pl.plot(res['t'], y, label=label, **plot_args, c=this_color)
                if key in data_mapping:
                    pl.scatter(self.data['day'], data_mapping[key], c=[this_color], **scatter_args)
            if self.data is not None and len(self.data):
                pl.scatter(pl.nan, pl.nan, c=[(0,0,0)], label='Data', **scatter_args)
            pl.grid(use_grid)
            cova.fixaxis(self)
            sc.commaticks()
            pl.title(title)

            # Set xticks as dates # TODO: make more general-purpose!
            ax = pl.gca()
            xmin,xmax = ax.get_xlim()
            ax.set_xticks(pl.arange(xmin, xmax+1, 7))
            xt = ax.get_xticks()
            lab = []
            for t in xt:
                tmp = self['day_0'] + dt.timedelta(days=int(t)) # + pars['day_0']
                lab.append(tmp.strftime('%B %d'))
            ax.set_xticklabels(lab)

        # Ensure the figure actually renders or saves
        if do_save:
            if isinstance(do_save, str) and fig_path is None:
                fig_path = do_save # It's a string, assume it's a filename
            else:
                fig_path = 'covasim.png' # Just give it a default name
            pl.savefig(fig_path)

        pl.show()

        return fig


    def plot_people(self):
        ''' Use imshow() to show all individuals as rows, with time as columns, one pixel per timestep per person '''
        raise NotImplementedError
