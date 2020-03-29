'''
Defines the Sim class, Covasim's core class.
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import pylab as pl
import sciris as sc
import datetime as dt
from . import utils as cvu
from . import base as cvbase
from . import parameters as cvpars
from . import people as cvppl


# Specify all externally visible functions this file defines
__all__ = ['default_sim_plots', 'Sim']


# Specify which quantities to plot -- note, these can be turned on and off by commenting/uncommenting lines
default_sim_plots = sc.odict({
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


class Sim(cvbase.BaseSim):
    '''
    The Sim class handles the running of the simulation: the number of children,
    number of time points, and the parameters of the simulation.
    '''

    def __init__(self, pars=None, datafile=None, filename=None):
        default_pars = cvpars.make_pars() # Start with default pars
        super().__init__(default_pars) # Initialize and set the parameters as attributes
        self.datafile = datafile # Store this
        self.data = None
        if datafile is not None: # If a data file is provided, load it
            self.data = cvpars.load_data(datafile)
        self.created = sc.now()
        if filename is None:
            datestr = sc.getdate(obj=self.created, dateformat='%Y-%b-%d_%H.%M.%S')
            filename = f'covasim_{datestr}.sim'
        self.filename = filename
        self.stopped = None # If the simulation has stopped
        self.results_ready = False # Whether or not results are ready
        self.people = {}
        self.results = {}
        self.calculated = {}
        if pars is not None:
            self.update_pars(pars)
        return


    def initialize(self):
        ''' Perform all initializations '''
        self.validate_pars()
        self.set_seed()
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
            start_day = sc.readdate(start_day)
        self['start_day'] = start_day # Convert back

        # Handle population data
        popdata_choices = ['random', 'bayesian', 'data']
        if sc.isnumber(self['usepopdata']) or isinstance(self['usepopdata'], bool): # Convert e.g. usepopdata=1 to 'bayesian'
            self['usepopdata'] = popdata_choices[int(self['usepopdata'])] # Choose one of these
        if self['usepopdata'] not in popdata_choices:
            choice = self['usepopdata']
            choicestr = ', '.join(popdata_choices)
            errormsg = f'Population data option "{choice}" not available; choices are: {choicestr}'
            raise ValueError(errormsg)

        # Handle interventions
        self['interventions'] = sc.promotetolist(self['interventions'], keepnone=False)

        return


    def init_results(self):
        ''' Initialize results '''

        def init_res(*args, **kwargs):
            ''' Initialize a single result object '''
            output = cvbase.Result(*args, **kwargs, npts=self.npts)
            return output

        # Create the main results structure
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

        # Populate the rest of the results
        self.results['t'] = self.tvec
        self.results['date'] = [self['start_day'] + dt.timedelta(days=int(t)) for t in self.tvec]
        self.results_ready = False

        # Create calculated values structure
        self.calculated['eff_beta'] = (1-self['default_symp_prob'])*self['asym_factor']*self['beta'] + self['default_symp_prob']*self['beta']  # Using asymptomatic proportion
        self.calculated['r_0']      = self['contacts']*self['dur']*self.calculated['eff_beta']
        return


    @property
    def reskeys(self):
        ''' Get the actual results objects, not other things stored in sim.results '''
        all_keys = list(self.results.keys())
        res_keys = []
        for key in all_keys:
            if isinstance(self.results[key], cvbase.Result):
                res_keys.append(key)
        return res_keys


    def init_people(self, verbose=None, id_len=None):
        ''' Create the people '''
        if verbose is None:
            verbose = self['verbose']

        sc.printv(f'Creating {self["n"]} people...', 1, verbose)

        cvppl.make_people(self, verbose=verbose, id_len=id_len)

        # Create the seed infections
        for i in range(int(self['n_infected'])):
            person = self.get_person(i)
            person.infect(t=0)

        return


    def run(self, initialize=True, do_plot=False, verbose=None, **kwargs):
        ''' Run the simulation '''

        T = sc.tic()

        # Reset settings and results
        if verbose is None:
            verbose = self['verbose']
        if initialize:
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
                self.stopped = self['stop_func'](self, t) # Feed in the current simulation object and the time

            # If this gets set, stop running -- e.g. if the time limit is exceeded
            if self.stopped:
                break

            # Zero counts for this time step.
            n_susceptible = 0
            n_exposed     = 0
            n_deaths      = 0
            n_recoveries  = 0
            n_infectious  = 0
            n_infections  = 0
            n_symptomatic = 0
            n_recovered   = 0

            # Extract these for later use. The values do not change in the person loop and the dictionary lookup is expensive.
            rand_popdata     = (self['usepopdata'] == 'random')
            beta             = self['beta']
            asym_factor      = self['asym_factor']
            diag_factor      = self['diag_factor']
            cont_factor      = self['cont_factor']
            beta_pop         = self['beta_pop']

            # Print progress
            if verbose>=1:
                string = f'  Running day {t:0.0f} of {self.pars["n_days"]} ({elapsed:0.2f} s elapsed)...'
                if verbose>=2:
                    sc.heading(string)
                else:
                    print(string)


            # Update each person, skipping people who are susceptible
            not_susceptible = filter(lambda p: not p.susceptible, self.people.values())
            n_susceptible = len(self.people)

            for person in not_susceptible:
                n_susceptible -= 1

                # If exposed, check if the person becomes infectious or develops symptoms
                if person.exposed:
                    n_exposed += 1
                    if not person.infectious and t >= person.date_infectious: # It's the day they become infectious
                        person.infectious = True
                        sc.printv(f'      Person {person.uid} became infectious!', 2, verbose)
                    if not person.symptomatic and person.date_symptomatic is not None and t >= person.date_symptomatic:  # It's the day they develop symptoms
                        person.symptomatic = True
                        sc.printv(f'      Person {person.uid} developed symptoms!', 2, verbose)

                # If infectious, check if anyone gets infected
                if person.infectious:

                    # Check for death
                    died = person.check_death(t)
                    n_deaths += died

                    # Check for recovery
                    recovered = person.check_recovery(t)
                    n_recoveries += recovered

                    # If the person didn't die or recover, check for onward transmission
                    if not died and not recovered:
                        n_infectious += 1 # Count this person as infectious

                        # Calculate transmission risk based on whether they're asymptomatic/diagnosed/have been isolated
                        thisbeta = beta * \
                                   (asym_factor if person.symptomatic else 1.) * \
                                   (diag_factor if person.diagnosed else 1.) * \
                                   (cont_factor if person.known_contact else 1.)

                        # Determine who gets infected
                        if rand_popdata: # Flat contacts
                            transmission_inds = cvu.bf(thisbeta, person.contacts)
                        else: # Dictionary of contacts -- extra loop over layers
                            transmission_inds = []
                            for ckey in self.contact_keys:
                                layer_beta = thisbeta * beta_pop[ckey]
                                transmission_inds.extend(cvu.bf(layer_beta, person.contacts[ckey]))

                        # Loop over people who do
                        for contact_ind in transmission_inds:
                            target_person = self.get_person(contact_ind) # Stored by integer

                            # This person was diagnosed last time step: time to flag their contacts
                            if person.date_diagnosed is not None and person.date_diagnosed == t-1:
                                target_person.known_contact = True

                            # Skip people who are not susceptible
                            if target_person.susceptible:
                                n_infections += target_person.infect(t, person) # Actually infect them
                                sc.printv(f'        Person {person.uid} infected person {target_person.uid}!', 2, verbose)


                # Count people who developed symptoms
                if person.symptomatic:
                    n_symptomatic += 1

                # Count people who recovered
                if person.recovered:
                    n_recovered += 1

            # End of person loop; apply interventions
            for intervention in self['interventions']:
                intervention.apply(self, t)
            if self['interv_func'] is not None: # Apply custom intervention function
                self =self['interv_func'](self, t)

            # Update counts for this time step
            self.results['n_susceptible'][t] = n_susceptible
            self.results['n_exposed'][t]     = n_exposed
            self.results['deaths'][t]        = n_deaths
            self.results['recoveries'][t]    = n_recoveries
            self.results['n_infectious'][t]  = n_infectious
            self.results['infections'][t]    = n_infections
            self.results['n_symptomatic'][t] = n_symptomatic
            self.results['n_recovered'][t]   = n_recovered

        # End of time loop; compute cumulative results outside of the time loop
        self.results['cum_exposed'].values    = pl.cumsum(self.results['infections'].values) + self['n_infected'] # Include initially infected people
        self.results['cum_tested'].values     = pl.cumsum(self.results['tests'].values)
        self.results['cum_diagnosed'].values  = pl.cumsum(self.results['diagnoses'].values)
        self.results['cum_deaths'].values     = pl.cumsum(self.results['deaths'].values)
        self.results['cum_recoveries'].values = pl.cumsum(self.results['recoveries'].values)

        # Add in the results from the interventions
        for intervention in self['interventions']:
            intervention.finalize(self)  # Execute any post-processing

        # Scale the results
        for reskey in self.reskeys:
            if self.results[reskey].scale:
                self.results[reskey].values *= self['scale']

        # Perform calculations on results
        self.compute_doubling()
        self.compute_r_eff()
        self.likelihood()

        # Tidy up
        self.results_ready = True
        sc.printv(f'\nRun finished after {elapsed:0.1f} s.\n', 1, verbose)
        self.results['summary'] = self.summary_stats()

        if do_plot:
            self.plot(**kwargs)

        # Convert to an odict to allow e.g. sim.people[25] later, and results to an objdict to allow e.g. sim.results.diagnoses
        self.people = sc.odict(self.people)
        self.results = sc.objdict(self.results)

        return self.results


    def compute_doubling(self, window=None, max_doubling_time=100):
        '''
        Calculate doubling time using exponential approximation -- a more detailed
        approach is in utils.py. Compares infections at time t to infections at time
        t-window, and uses that to compute the doubling time. For example, if there are
        100 cumulative infections on day 12 and 200 infections on day 19, doubling
        time is 7 days.

        Args:
            window (float): the size of the window used (larger values are more accurate but less precise)
            max_doubling_time (float): doubling time could be infinite, so this places a bound on it

        Returns:
            None (modifies results in place)
        '''
        window = self['window']
        cum_infections = self.results['cum_exposed']
        for t in range(window, self.npts):
            infections_now = cum_infections[t]
            infections_prev = cum_infections[t-window]
            r = infections_now/infections_prev
            if r > 1:  # Avoid divide by zero
                doubling_time = window*np.log(2)/np.log(r)
                doubling_time = min(doubling_time, max_doubling_time) # Otherwise, it's unbounded
                self.results['doubling_time'][t] = doubling_time
        return


    def compute_r_eff(self):
        ''' Effective reproductive number based on number still susceptible -- TODO: use data instead '''

        # Initialize arrays to hold sources and targets infected each day
        sources = np.zeros(self.npts)
        targets = np.zeros(self.npts)

        # Loop over each person to pull out the transmission
        for person in self.people.values():
            if person.date_exposed is not None: # Skip people who were never exposed
                if person.date_recovered is not None:
                    outcome_date = person.date_recovered
                elif person.date_died is not None:
                    outcome_date = person.date_died
                else:
                    errormsg = f'No outcome (death or recovery) can be determined for the following person:\n{person}'
                    raise ValueError(errormsg)

                if outcome_date is not None and outcome_date<self.npts:
                    outcome_date = int(outcome_date)
                    sources[outcome_date] += 1
                    targets[outcome_date] += len(person.infected)

        # Populate the array -- to avoid divide-by-zero, skip indices that are 0
        inds = sc.findinds(sources>0)
        r_eff = targets[inds]/sources[inds]
        self.results['r_eff'].values[inds] = r_eff
        return


    def likelihood(self, verbose=None):
        '''
        Compute the log-likelihood of the current simulation based on the number
        of new diagnoses.
        '''
        if verbose is None:
            verbose = self['verbose']

        loglike = 0
        if self.data is not None and len(self.data): # Only perform likelihood calculation if data are available
            for d,datum in enumerate(self.data['new_positives']):
                if not pl.isnan(datum): # Skip days when no tests were performed
                    estimate = self.results['diagnoses'][d]
                    p = cvu.poisson_test(datum, estimate)
                    logp = pl.log(p)
                    loglike += logp
                    sc.printv(f'  {self.data["date"][d]}, data={datum:3.0f}, model={estimate:3.0f}, log(p)={logp:10.4f}, loglike={loglike:10.4f}', 2, verbose)

            self.results['likelihood'] = loglike
            sc.printv(f'Likelihood: {loglike}', 1, verbose)

        return loglike


    def summary_stats(self, verbose=None):
        ''' Compute the summary statistics to display at the end of a run '''

        if verbose is None:
            verbose = self['verbose']

        summary = sc.objdict()
        for key in self.reskeys:
            summary[key] = self.results[key][-1]

        sc.printv(f"""Summary:
     {summary['n_susceptible']:5.0f} susceptible
     {summary['n_infectious']:5.0f} infectious
     {summary['n_symptomatic']:5.0f} symptomatic
     {summary['cum_exposed']:5.0f} total exposed
     {summary['cum_diagnosed']:5.0f} total diagnosed
     {summary['cum_deaths']:5.0f} total deaths
     {summary['cum_recoveries']:5.0f} total recovered
               """, 1, verbose)

        return summary


    def plot(self, to_plot=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
             scatter_args=None, axis_args=None, as_dates=True, interval=None, dateformat=None,
             font_size=18, font_family=None, use_grid=True, use_commaticks=True, do_show=True,
             verbose=None):
        '''
        Plot the results -- can supply arguments for both the figure and the plots.

        Args:
            to_plot (dict): Nested dict of results to plot; see default_sim_plots for structure
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
            use_commaticks (bool): Plot y-axis with commas rather than scientific notation
            do_show (bool): Whether or not to show the figure
            verbose (bool): Display a bit of extra information

        Returns:
            fig: Figure handle
        '''

        if verbose is None:
            verbose = self['verbose']
        sc.printv('Plotting...', 1, verbose)

        if to_plot is None:
            to_plot = default_sim_plots
        to_plot = sc.odict(to_plot) # In case it's supplied as a dict

        # Handle input arguments -- merge user input with defaults
        fig_args     = sc.mergedicts({'figsize':(16,12)}, fig_args)
        plot_args    = sc.mergedicts({'lw':3, 'alpha':0.7}, plot_args)
        scatter_args = sc.mergedicts({'s':150, 'marker':'s'}, scatter_args)
        axis_args    = sc.mergedicts({'left':0.1, 'bottom':0.05, 'right':0.9, 'top':0.97, 'wspace':0.2, 'hspace':0.25}, axis_args)

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
            cvu.fixaxis(self)
            if use_commaticks:
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

            # Plot interventions
            for intervention in self['interventions']:
                intervention.plot(self, ax)

        # Ensure the figure actually renders or saves
        if do_save:
            if fig_path is None: # No figpath provided - see whether do_save is a figpath
                if isinstance(do_save, str) :
                    fig_path = do_save # It's a string, assume it's a filename
                else:
                    fig_path = 'covasim.png' # Just give it a default name
            fig_path = sc.makefilepath(fig_path) # Ensure it's valid, including creating the folder
            pl.savefig(fig_path)

        if do_show:
            pl.show()
        else:
            pl.close(fig)

        return fig
