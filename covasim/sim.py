'''
Defines the Sim class, Covasim's core class.
'''

#%% Imports
import numpy as np
import pylab as pl
import sciris as sc
import datetime as dt
import matplotlib.ticker as ticker
from . import version as cvv
from . import utils as cvu
from . import defaults as cvd
from . import base as cvbase
from . import parameters as cvpars
from . import population as cvpop

# Specify all externally visible things this file defines
__all__ = ['Sim']

class Sim(cvbase.BaseSim):
    '''
    The Sim class handles the running of the simulation: the number of children,
    number of time points, and the parameters of the simulation.

    Args:
        pars (dict): parameters to modify from their default values
        datafile (str): filename of (Excel) data file to load, if any
        datacols (list): list of column names of the data file to load
        filename (str): the filename for this simulation, if it's saved (default: creation date)
    '''

    def __init__(self, pars=None, datafile=None, datacols=None, popfile=None, filename=None, **kwargs):
        # Create the object
        default_pars = cvpars.make_pars(**kwargs) # Start with default pars
        super().__init__(default_pars) # Initialize and set the parameters as attributes

        # Set attributes
        self.created       = None  # The datetime the sim was created
        self.filename      = None  # The filename of the sim
        self.datafile      = None  # The name of the data file
        self.data          = None  # The actual data
        self.popdict       = None  # The population dictionary
        self.t             = None  # The current time in the simulation
        self.initialized   = False # Whether or not initialization is complete
        self.results_ready = False # Whether or not results are ready
        self.people        = []    # Initialize these here so methods that check their length can see they're empty
        self.contact_keys  = None  # Keys for contact networks
        self.results       = {}    # For storing results

        # Now update everything
        self.set_metadata(filename)        # Set the simulation date and filename
        self.load_data(datafile, datacols) # Load the data, if provided
        self.load_population(popfile)      # Load the population, if provided
        self.update_pars(pars)             # Update the parameters, if provided

        return


    def update_pars(self, pars=None, create=False, **kwargs):
        ''' Ensure that metaparameters get used properly before being updated '''
        pars = sc.mergedicts(pars, kwargs)
        if pars:
            if 'use_layers' in pars: # Reset layers
                cvpars.set_contacts(pars)
            if 'prog_by_age' in pars:
                pars['prognoses'] = cvpars.get_prognoses(by_age=pars['prog_by_age']) # Reset prognoses
            super().update_pars(pars=pars, create=create) # Call update_pars() for ParsObj
        return


    def set_metadata(self, filename):
        ''' Set the metadata for the simulation -- creation time and filename '''
        self.created = sc.now()
        self.version = cvv.__version__
        self.git_info = cvu.git_info()
        if filename is None:
            datestr = sc.getdate(obj=self.created, dateformat='%Y-%b-%d_%H.%M.%S')
            self.filename = f'covasim_{datestr}.sim'
        return


    def load_data(self, datafile=None, datacols=None, **kwargs):
        ''' Load the data to calibrate against, if provided '''
        self.datafile = datafile # Store this
        if datafile is not None: # If a data file is provided, load it
            self.data = cvpars.load_data(filename=datafile, columns=datacols, **kwargs)

            # Ensure the data are continuous and align with the simulation
            # data_offset = (self.data.iloc[0]['date'] - self.pars['start_day']).days # TODO: Use df.set_index("A").reindex(new_index).reset_index()

        return


    def load_population(self, filename=None, **kwargs):
        '''
        Load the population dictionary from file.

        Args:
            filename (str): name of the file to load
        '''
        if filename is not None:
            filepath = sc.makefilepath(filename=filename, **kwargs)
            self.popdict = sc.loadobj(filepath)
            n_actual = len(self.popdict['uid'])
            n_expected = self['pop_size']
            if n_actual != n_expected:
                errormsg = f'Wrong number of people ({n_expected} requested, {n_actual} actual) -- please change "pop_size" to match or regenerate the file'
                raise ValueError(errormsg)
        return


    def save_population(self, filename, **kwargs):
        '''
        Save the population dictionary to file.

        Args:
            filename (str): name of the file to save to.
        '''
        filepath = sc.makefilepath(filename=filename, **kwargs)
        sc.saveobj(filepath, self.popdict)
        return filepath


    def initialize(self, **kwargs):
        '''
        Perform all initializations.

        Args:
            kwargs (dict): passed to init_people
        '''
        self.t = 0  # The current time index
        self.validate_pars() # Ensure parameters have valid values
        self.set_seed() # Reset the random seed
        self.init_results() # Create the results stucture
        self.init_people(**kwargs) # Create all the people (slow)
        self.initialized = True
        return


    def validate_pars(self):
        ''' Some parameters can take multiple types; this makes them consistent '''

        # Handle start day
        start_day = self['start_day'] # Shorten
        if start_day in [None, 0]: # Use default start day
            start_day = dt.date(2020, 1, 1)
        elif sc.isstring(start_day):
            start_day = sc.readdate(start_day)
        if isinstance(start_day,dt.datetime):
            start_day = start_day.date()
        self['start_day'] = start_day

        # Handle contacts
        contacts = self['contacts']
        if sc.isnumber(contacts): # It's a scalar instead of a dict, assume it's all contacts
            self['contacts']    = {'a':contacts}
            self['beta_layers'] = {'a':1.0}

        # Handle population data
        popdata_choices = ['random', 'hybrid', 'clustered', 'synthpops']
        if sc.isnumber(self['pop_type']): # Convert e.g. pop_type=1 to 'hybrid'
            self['pop_type'] = popdata_choices[int(self['pop_type'])] # Choose one of these
        if self['pop_type'] not in popdata_choices:
            choice = self['pop_type']
            choicestr = ', '.join(popdata_choices)
            errormsg = f'Population type "{choice}" not available; choices are: {choicestr}'
            raise ValueError(errormsg)

        # Handle interventions
        self['interventions'] = sc.promotetolist(self['interventions'], keepnone=False)

        return


    def init_results(self):
        '''
        Create the main results structure.
        We differentiate between flows, stocks, and cumulative results
        The prefix "new" is used for flow variables, i.e. counting new events (infections/deaths/recoveries) on each timestep
        The prefix "n" is used for stock variables, i.e. counting the total number in any given state (sus/inf/rec/etc) on any paticular timestep
        The prefix "cum_" is used for cumulative variables, i.e. counting the total number that have ever been in a given state at some point in the sim
        Note that, by definition, n_dead is the same as cum_deaths and n_recovered is the same as cum_recoveries, so we only define the cumulative versions
        '''

        def init_res(*args, **kwargs):
            ''' Initialize a single result object '''
            output = cvbase.Result(*args, **kwargs, npts=self.npts)
            return output

        dcols = cvd.default_colors # Shorten default colors

        # Stock variables
        for key,label in cvd.result_stocks.items():
            self.results[f'n_{key}'] = init_res(label, color=dcols[key])
        self.results['n_susceptible'].scale = 'static'
        self.results['bed_capacity']  = init_res('Percentage bed capacity', scale=False)

        # Flows and cumulative flows
        for key,label in cvd.result_flows.items():
            self.results[f'new_{key}'] = init_res(f'Number of new {label}', color=dcols[key]) # Flow variables -- e.g. "Number of new infections"
            self.results[f'cum_{key}'] = init_res(f'Cumulative {label}',    color=dcols[key]) # Cumulative variables -- e.g. "Cumulative infections"

        # Other variables
        self.results['r_eff']         = init_res('Effective reproductive number', scale=False)
        self.results['doubling_time'] = init_res('Doubling time', scale=False)

        # Populate the rest of the results
        if self['rescale']:
            scale = 1
        else:
            scale = self['pop_scale']
        self.rescale_vec   = scale*np.ones(self.npts)
        self.results['t']    = self.tvec
        self.results['date'] = self.datevec
        self.results_ready   = False

        return


    @property
    def reskeys(self):
        ''' Get the actual results objects, not other things stored in sim.results '''
        all_keys = list(self.results.keys())
        res_keys = [key for key in all_keys if isinstance(self.results[key], cvbase.Result)]
        return res_keys


    def init_people(self, verbose=None, id_len=None, **kwargs):
        ''' Create the people '''

        if verbose is None:
            verbose = self['verbose']

        sc.printv(f'Creating {self["pop_size"]} people...', 1, verbose)

        cvpop.make_people(self, verbose=verbose, **kwargs)

        # Create the seed infections
        for i in range(int(self['pop_infected'])):
            person = self.people[i]
            person.infect(t=0)

        return


    def next(self, verbose=0):
        '''
        Step simulation forward in time
        '''

        # Set the time and if we have reached the end of the simulation, then do nothing
        t = self.t
        if t >= self.npts:
            return

        # Zero counts for this time step: stocks
        n_susceptible   = 0
        n_exposed       = 0
        n_infectious    = 0
        n_symptomatic   = 0
        n_severe        = 0
        n_critical      = 0
        n_diagnosed     = 0
        n_quarantined   = 0

        # Zero counts for this time step: flows
        new_recoveries  = 0
        new_deaths      = 0
        new_infections  = 0
        new_symptomatic = 0
        new_severe      = 0
        new_critical    = 0
        new_quarantined = 0

        # Extract these for later use. The values do not change in the person loop and the dictionary lookup is expensive.
        beta             = self['beta']
        asymp_factor     = self['asymp_factor']
        diag_factor      = self['diag_factor']
        quar_trans_factor= self['quar_trans_factor']
        quar_acq_factor  = self['quar_acq_factor']
        quar_period      = self['quar_period']
        beta_layers      = self['beta_layers']
        n_beds           = self['n_beds']
        bed_constraint   = False
        pop_size         = len(self.people)
        n_imports        = cvu.pt(self['n_imports']) # Imported cases
        if 'c' in self['contacts']:
            n_comm_contacts = self['contacts']['c'] # Community contacts; TODO: make less ugly
        else:
            n_comm_contacts = 0

        # Print progress
        if verbose >= 1:
            string = f'  Running day {t:0.0f} of {self.pars["n_days"]} ({sc.toc(output=True):0.2f} s elapsed)...'
            if verbose >= 2:
                sc.heading(string)
            else:
                print(string)

        # Check if we need to rescale
        if self['rescale']:
            self.rescale()

        # Randomly infect some people (imported infections)
        if n_imports>0:
            imporation_inds = cvu.choose(max_n=pop_size, n=n_imports)
            for ind in imporation_inds:
                person = self.people[ind]
                new_infections += person.infect(t=t)


        susceptible = self.people.filter_in('susceptible')
        n_susceptible = 0
        for person in susceptible:
            n_susceptible += 1 # Update number of susceptibles

            # If they're quarantined, this affects their transmission rate
            new_quarantined += person.check_quar_begin(t, quar_period) # Set know_contact and go into quarantine
            n_quarantined += person.check_quar_end(t) # Come out of quarantine, and count quarantine state

        # Loop over everyone not susceptible
        not_susceptible = self.people.filter_out('susceptible')
        for person in not_susceptible:
            # N.B. Recovered and dead people are included here!

            # If exposed, check if the person becomes infectious
            if person.exposed:
                n_exposed += 1
                if not person.infectious and t == person.date_infectious: # It's the day they become infectious
                    person.infectious = True
                    sc.printv(f'      Person {person.uid} became infectious!', 2, verbose)

            # If they're quarantined, this affects their transmission rate
            new_quarantined += person.check_quar_begin(t, quar_period) # Set know_contact and go into quarantine
            person.check_quar_end(t) # Come out of quarantine
            n_quarantined += person.quarantined
            n_diagnosed   += person.diagnosed

            # If infectious, update status according to the course of the infection, and check if anyone gets infected
            if person.infectious:

                # Check whether the person died on this timestep
                new_death = person.check_death(t)
                new_deaths += new_death

                # Check whether the person recovered on this timestep
                new_recovery = person.check_recovery(t)
                new_recoveries += new_recovery

                # If the person didn't die or recover, check for onward transmission
                if not new_death and not new_recovery:
                    n_infectious += 1 # Count this person as infectious

                    # Check symptoms and diagnosis
                    new_symptomatic += person.check_symptomatic(t)
                    new_severe      += person.check_severe(t)
                    new_critical    += person.check_critical(t)
                    n_symptomatic   += person.symptomatic
                    n_severe        += person.severe
                    n_critical      += person.critical
                    if n_severe > n_beds:
                        bed_constraint = True

                    # Calculate transmission risk based on whether they're asymptomatic/diagnosed/have been isolated
                    thisbeta = beta * \
                               (asymp_factor if not person.symptomatic else 1.) * \
                               (diag_factor if person.diagnosed else 1.)

                    # Set community contacts
                    person_contacts = person.contacts
                    if n_comm_contacts:
                        community_contact_inds = cvu.choose(max_n=pop_size, n=n_comm_contacts)
                        person_contacts['c'] = community_contact_inds

                    # Determine who gets infected
                    for ckey in self.contact_keys:
                        contact_ids = person_contacts[ckey]
                        if len(contact_ids):
                            this_beta_layer = thisbeta *\
                                              beta_layers[ckey] *\
                                              (quar_trans_factor[ckey] if person.quarantined else 1.) # Reduction in onward transmission due to quarantine

                            transmission_inds = cvu.bf(this_beta_layer, contact_ids)
                            for contact_ind in transmission_inds: # Loop over people who get infected
                                target_person = self.people[contact_ind]
                                if target_person.susceptible: # Skip people who are not susceptible

                                    # See whether we will infect this person
                                    infect_this_person = True # By default, infect them...
                                    if target_person.quarantined:
                                        infect_this_person = cvu.bt(quar_acq_factor) # ... but don't infect them if they're isolating # DJK - should be layer dependent!
                                    if infect_this_person:
                                        new_infections += target_person.infect(t, bed_constraint, source=person) # Actually infect them
                                        sc.printv(f'        Person {person.uid} infected person {target_person.uid}!', 2, verbose)

        # End of person loop; apply interventions
        for intervention in self['interventions']:
            intervention.apply(self)
        if self['interv_func'] is not None: # Apply custom intervention function
            self =self['interv_func'](self)

        # Update counts for this time step: stocks
        self.results['n_susceptible'][t]  = n_susceptible - new_infections
        self.results['n_exposed'][t]      = n_exposed
        self.results['n_infectious'][t]   = n_infectious # Tracks total number infectious at this timestep
        self.results['n_symptomatic'][t]  = n_symptomatic # Tracks total number symptomatic at this timestep
        self.results['n_severe'][t]       = n_severe # Tracks total number of severe cases at this timestep
        self.results['n_critical'][t]     = n_critical # Tracks total number of critical cases at this timestep
        self.results['n_diagnosed'][t]    = n_diagnosed # Tracks total number of diagnosed cases at this timestep
        self.results['n_quarantined'][t]   = n_quarantined # Tracks number currently quarantined
        self.results['bed_capacity'][t]   = n_severe/n_beds if n_beds>0 else np.nan

        # Update counts for this time step: flows
        self.results['new_infections'][t]  = new_infections # New infections on this timestep
        self.results['new_recoveries'][t]  = new_recoveries # Tracks new recoveries on this timestep
        self.results['new_symptomatic'][t] = new_symptomatic
        self.results['new_severe'][t]      = new_severe
        self.results['new_critical'][t]    = new_critical
        self.results['new_deaths'][t]      = new_deaths
        self.results['new_quarantined'][t] = new_quarantined

        self.t += 1


    def rescale(self):
        ''' Dynamically rescale the population '''
        t = self.t
        pop_scale = self['pop_scale']
        current_scale = self.rescale_vec[t]
        if current_scale < pop_scale: # We have room to rescale
            not_sus = list(self.people.filter_out('susceptible'))
            n_not_sus = len(not_sus)
            n_people = len(self.people)
            if n_not_sus / n_people > self['rescale_threshold']: # Check if we've reached point when we want to rescale
                max_ratio = pop_scale/current_scale # We don't want to exceed this
                scaling_ratio = min(self['rescale_factor'], max_ratio)
                self.rescale_vec[t+1:] *= scaling_ratio # Update the rescaling factor from here on
                n = int(n_people*(1.0-1.0/scaling_ratio)) # For example, rescaling by 2 gives n = 0.5*n_people
                new_susceptibles = cvu.choose(max_n=n_people, n=n) # Choose who to make susceptible again
                for p in new_susceptibles: # TODO: only loop over non-susceptibles
                    person = self.people[p]
                    if not person.susceptible:
                        person.make_susceptible()
        return


    def run(self, do_plot=False, verbose=None, **kwargs):
        '''
        Run the simulation.

        Args:
            do_plot (bool): whether to plot
            verbose (int): level of detail to print
            kwargs (dict): passed to self.plot()

        Returns:
            results: the results object (also modifies in-place)
        '''

        T = sc.tic()

        # Reset settings and results
        if not self.initialized:
            self.initialize()

        if verbose is None:
            verbose = self['verbose']

        # Main simulation loop
        for t in range(self.npts):

            # Do the heavy lifting
            self.next(verbose=verbose)

            # Check if we were asked to stop
            elapsed = sc.toc(T, output=True)
            if elapsed > self['timelimit']:
                sc.printv(f"Time limit ({self['timelimit']} s) exceeded", 1, verbose)
                break
            elif self['stopping_func'] and self['stopping_func'](self):
                sc.printv("Stopping function terminated the simulation", 1, verbose)
                break

        # End of time loop; compute cumulative results outside of the time loop
        self.finalize(verbose=verbose) # Finalize the results
        sc.printv(f'\nRun finished after {elapsed:0.1f} s.\n', 1, verbose)
        self.summary = self.summary_stats(verbose=verbose)
        if do_plot: # Optionally plot
            self.plot(**kwargs)

        return self.results


    def finalize(self, verbose=None):
        ''' Compute final results, likelihood, etc. '''

        # Scale the results
        for reskey in self.reskeys:
            if self.results[reskey].scale == 'dynamic':
                self.results[reskey].values *= self.rescale_vec
            elif self.results[reskey].scale == 'static':
                self.results[reskey].values *= self['pop_scale']

        # Calculate cumulative results
        for key in cvd.result_flows.keys():
            self.results[f'cum_{key}'].values = np.cumsum(self.results[f'new_{key}'].values)
        self.results['cum_infections'].values += self['pop_infected']*self.rescale_vec[0] # Include initially infected people

        # Perform calculations on results
        self.compute_doubling()
        self.compute_r_eff()
        self.likelihood()

        # Convert results to a odicts/objdict to allow e.g. sim.results.diagnoses
        # self.people = sc.odict({str(p):person for p,person in enumerate(self.people)}) # Convert to an odict for a better repr
        self.results = sc.objdict(self.results)
        self.results_ready = True

        return


    def compute_doubling(self, window=7, max_doubling_time=50):
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
        cum_infections = self.results['cum_infections'].values
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
        '''
        Effective reproductive number based on number of people each person infected.
        '''

        # Initialize arrays to hold sources and targets infected each day
        sources = np.zeros(self.npts)
        targets = np.zeros(self.npts)

        # Loop over each person to pull out the transmission
        for person in self.people:
            if person.date_exposed is not None: # Skip people who were never exposed
                if person.date_recovered is not None:
                    outcome_date = person.date_recovered
                elif person.date_dead is not None:
                    outcome_date = person.date_dead
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


    def compute_gen_time(self):
        '''
        Calculate the generation time (or serial interval) there are two
        ways to do this calculation. The 'true' interval (exposure time to
        exposure time) or 'clinical' (symptom onset to symptom onset).
        '''

        not_susceptible = self.people.filter_out('susceptible')
        intervals = np.zeros(int(self.summary['cum_infections']))
        intervals2 = intervals.copy()
        pos = 0
        pos2 = 0
        for source in not_susceptible:
            if len(source.infected)>0:
                for target in source.infected:
                    intervals[pos] = self.people[target].date_exposed - source.date_exposed
                    pos += 1
                if source.date_symptomatic is not None:
                    for target in source.infected:
                        if self.people[target].date_symptomatic is not None:
                            intervals2[pos2] = self.people[target].date_symptomatic - source.date_symptomatic
                            pos2 += 1

        self.results['gen_time'] = {
                'true':         np.mean(intervals[:pos]),
                'true_std':     np.std(intervals[:pos]),
                'clinical':     np.mean(intervals2[:pos2]),
                'clinical_std': np.std(intervals2[:pos2])}
        return

    def likelihood(self, weights=None, verbose=None) -> float:
        '''
        Compute the log-likelihood of the current simulation based on the number
        of new diagnoses.
        '''

        if verbose is None:
            verbose = self['verbose']

        if weights is None:
            weights = {}

        if self.data is None:
            return np.nan

        loglike = 0

        model_dates = self.datevec.tolist()

        for key in set(self.reskeys).intersection(self.data.columns): # For keys present in both the results and in the data
            weight = weights.get(key, 1) # Use the provided weight if present, otherwise default to 1
            for d, datum in self.data[key].iteritems():
                if np.isfinite(datum):
                    if d in model_dates:
                        estimate = self.results[key][model_dates.index(d)]
                        if datum and estimate:
                            if (datum == 0) and (estimate == 0):
                                p = 1.0
                            else:
                                p = cvu.poisson_test(datum, estimate)
                            logp = pl.log(p)
                            loglike += weight*logp
                            sc.printv(f'  {d}, data={datum:3.0f}, model={estimate:3.0f}, log(p)={logp:10.4f}, loglike={loglike:10.4f}', 2, verbose)

            self.results['likelihood'] = loglike

        sc.printv(f'Likelihood: {loglike}', 1, verbose)
        return loglike


    def summary_stats(self, verbose=None):
        ''' Compute the summary statistics to display at the end of a run '''

        if verbose is None:
            verbose = self['verbose']

        summary = sc.objdict()
        summary_str = 'Summary:\n'
        for key in self.reskeys:
            summary[key] = self.results[key][-1]
            if key.startswith('cum_'):
                summary_str += f'   {summary[key]:5.0f} {self.results[key].name.lower()}\n'
        sc.printv(summary_str, 1, verbose)

        return summary


    def plot(self, to_plot=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
             scatter_args=None, axis_args=None, legend_args=None, as_dates=True, dateformat=None,
             interval=None, n_cols=1, font_size=18, font_family=None, use_grid=True, use_commaticks=True,
             do_show=True, verbose=None):
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
            legend_args (dict): Dictionary of kwargs to be passed to pl.legend()
            as_dates (bool): Whether to plot the x-axis as dates or time points
            dateformat (str): Date string format, e.g. '%B %d'
            interval (int): Interval between tick marks
            n_cols (int): Number of columns of subpanels to use for subplot
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
            to_plot = cvd.default_sim_plots
        to_plot = sc.odict(to_plot) # In case it's supplied as a dict

        # Handle input arguments -- merge user input with defaults
        fig_args     = sc.mergedicts({'figsize':(16,14)}, fig_args)
        plot_args    = sc.mergedicts({'lw':3, 'alpha':0.7}, plot_args)
        scatter_args = sc.mergedicts({'s':70, 'marker':'s'}, scatter_args)
        axis_args    = sc.mergedicts({'left':0.1, 'bottom':0.05, 'right':0.9, 'top':0.97, 'wspace':0.2, 'hspace':0.25}, axis_args)
        legend_args  = sc.mergedicts({'loc': 'best'}, legend_args)

        fig = pl.figure(**fig_args)
        pl.subplots_adjust(**axis_args)
        pl.rcParams['font.size'] = font_size
        if font_family:
            pl.rcParams['font.family'] = font_family

        res = self.results # Shorten since heavily used

        # Plot everything
        n_rows = np.ceil(len(to_plot)/n_cols) # Number of subplot rows to have
        for p,title,keylabels in to_plot.enumitems():
            ax = pl.subplot(n_rows, n_cols, p+1)
            for key in keylabels:
                label = res[key].name
                this_color = res[key].color
                y = res[key].values
                pl.plot(res['t'], y, label=label, **plot_args, c=this_color)
                if self.data is not None and key in self.data:
                    data_t = (self.data.index-self['start_day'])/np.timedelta64(1,'D') # Convert from data date to model output index based on model start date
                    pl.scatter(data_t, self.data[key], c=[this_color], **scatter_args)
            if self.data is not None and len(self.data):
                pl.scatter(pl.nan, pl.nan, c=[(0,0,0)], label='Data', **scatter_args)

            pl.legend(**legend_args)
            pl.grid(use_grid)
            sc.setylim()
            if use_commaticks:
                sc.commaticks()
            pl.title(title)

            # Optionally reset tick marks (useful for e.g. plotting weeks/months)
            if interval:
                xmin,xmax = ax.get_xlim()
                ax.set_xticks(pl.arange(xmin, xmax+1, interval))

            # Set xticks as dates
            if as_dates:
                @ticker.FuncFormatter
                def date_formatter(x, pos):
                    return (self['start_day'] + dt.timedelta(days=x)).strftime('%b-%d')
                ax.xaxis.set_major_formatter(date_formatter)
                if not interval:
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

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


    def plot_result(self, key, fig_args=None, plot_args=None):
        '''
        Simple method to plot a single result. Useful for results that aren't
        standard outputs.

        Args:
            key (str): the key of the result to plot
            fig_args (dict): passed to pl.figure()
            plot_args (dict): passed to pl.plot()

        Example:
            sim.plot_result('doubling_time')
        '''
        fig_args  = sc.mergedicts({'figsize':(16,10)}, fig_args)
        plot_args = sc.mergedicts({'lw':3, 'alpha':0.7}, plot_args)
        fig = pl.figure(**fig_args)
        pl.subplot(111)
        tvec = self.results['t']
        res = self.results[key]
        y = res.values
        color = res.color
        pl.plot(tvec, y, c=color, **plot_args)
        return fig
