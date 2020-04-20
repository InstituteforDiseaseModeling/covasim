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
from . import misc as cvm
from . import defaults as cvd
from . import base as cvb
from . import parameters as cvpars
from . import population as cvpop

# Specify all externally visible things this file defines
__all__ = ['Sim']


class Sim(cvb.BaseSim):
    '''
    The Sim class handles the running of the simulation: the number of children,
    number of time points, and the parameters of the simulation.

    Args:
        pars (dict): parameters to modify from their default values
        datafile (str): filename of (Excel) data file to load, if any
        datacols (list): list of column names of the data file to load
        label (str): the name of the simulation (useful to distinguish in batch runs)
        simfile (str): the filename for this simulation, if it's saved (default: creation date)
        popfile (str): the filename to load/save the population for this simulation
        load_pop (bool): whether or not to load the population from the named file
        kwargs (dict): passed to make_pars()

    **Examples**::

        sim = cv.Sim()
        sim = cv.Sim(pop_size=10e3, datafile='my_data.xlsx')
    '''

    def __init__(self, pars=None, datafile=None, datacols=None, label=None, simfile=None, popfile=None, load_pop=False, **kwargs):
        # Create the object
        default_pars = cvpars.make_pars(**kwargs) # Start with default pars
        super().__init__(default_pars) # Initialize and set the parameters as attributes

        # Set attributes
        self.label         = label    # The label/name of the simulation
        self.created       = None     # The datetime the sim was created
        self.simfile       = simfile  # The filename of the sim
        self.datafile      = datafile # The name of the data file
        self.popfile       = popfile  # The population file
        self.data          = None     # The actual data
        self.popdict       = None     # The population dictionary
        self.t             = None     # The current time in the simulation
        self.people        = None     # Initialize these here so methods that check their length can see they're empty
        self.results       = {}       # For storing results
        self.initialized   = False    # Whether or not initialization is complete
        self.results_ready = False    # Whether or not results are ready

        # Now update everything
        self.set_metadata(simfile, label) # Set the simulation date and filename
        self.load_data(datafile, datacols) # Load the data, if provided
        self.update_pars(pars)             # Update the parameters, if provided
        if load_pop:
            self.load_population(popfile)      # Load the population, if provided

        return


    def update_pars(self, pars=None, create=False, **kwargs):
        ''' Ensure that metaparameters get used properly before being updated '''
        pars = sc.mergedicts(pars, kwargs)
        if pars:
            if 'pop_type' in pars:
                cvpars.reset_layer_pars(pars)
            if 'prog_by_age' in pars:
                pars['prognoses'] = cvpars.get_prognoses(by_age=pars['prog_by_age']) # Reset prognoses
            super().update_pars(pars=pars, create=create) # Call update_pars() for ParsObj
        return


    def set_metadata(self, simfile, label):
        ''' Set the metadata for the simulation -- creation time and filename '''
        self.created = sc.now()
        self.version = cvv.__version__
        self.git_info = cvm.git_info()
        if simfile is None:
            datestr = sc.getdate(obj=self.created, dateformat='%Y-%b-%d_%H.%M.%S')
            self.simfile = f'covasim_{datestr}.sim'
        if label is not None:
            self.label = label
        return


    def load_data(self, datafile=None, datacols=None, **kwargs):
        ''' Load the data to calibrate against, if provided '''
        self.datafile = datafile # Store this
        if datafile is not None: # If a data file is provided, load it
            self.data = cvpars.load_data(filename=datafile, columns=datacols, **kwargs)

        return


    def initialize(self, save_pop=False, load_pop=False, popfile=None, **kwargs):
        '''
        Perform all initializations.

        Args:
            save_pop (bool): if true, save the population to popfile
            load_pop (bool): if true, load the population from popfile
            popfile (str): filename to load/save the population
            kwargs (dict): passed to init_people
        '''
        self.t = 0  # The current time index
        self.validate_pars() # Ensure parameters have valid values
        self.set_seed() # Reset the random seed
        self.init_results() # Create the results stucture
        self.init_people(save_pop=save_pop, load_pop=load_pop, popfile=popfile, **kwargs) # Create all the people (slow)
        self.initialized = True
        return


    def reset_layer_pars(self, force=True):
        '''
        Reset the parameters to match the population.

        Args:
            force (bool): reset the pars even if they already exist
        '''
        if self.people is not None:
            layer_keys = self.people.contacts.keys()
        else:
            layer_keys = None
        cvpars.reset_layer_pars(self.pars, layer_keys=layer_keys, force=force)
        return


    def validate_pars(self):
        ''' Some parameters can take multiple types; this makes them consistent '''

        # Handle types
        for key in ['pop_size', 'pop_infected', 'pop_size', 'n_days']:
            self[key] = int(self[key])

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

        # Handle key mismaches
        beta_layer_keys = set(self.pars['beta_layer'].keys())
        contacts_keys   = set(self.pars['contacts'].keys())
        quar_eff_keys   = set(self.pars['quar_eff'].keys())
        if not(beta_layer_keys == contacts_keys == quar_eff_keys):
            errormsg = f'Layer parameters beta={beta_layer_keys}, contacts={contacts_keys}, quar_eff={quar_eff_keys} have inconsistent keys'
            raise cvm.KeyNotFoundError(errormsg)
        if self.people is not None:
            pop_keys = set(self.people.contacts.keys())
            if pop_keys != beta_layer_keys:
                errormsg = f'Please update your parameter keys {beta_layer_keys} to match population keys {pop_keys}. You may find sim.reset_layer_pars() helpful.'
                raise cvm.KeyNotFoundError(errormsg)

        # Handle population data
        popdata_choices = ['random', 'hybrid', 'clustered', 'synthpops']
        choice = self['pop_type']
        if choice not in popdata_choices:
            choicestr = ', '.join(popdata_choices)
            errormsg = f'Population type "{choice}" not available; choices are: {choicestr}'
            raise cvm.KeyNotFoundError(errormsg)

        # Handle interventions
        self['interventions'] = sc.promotetolist(self['interventions'], keepnone=False)

        return


    def init_results(self):
        '''
        Create the main results structure.
        We differentiate between flows, stocks, and cumulative results
        The prefix "new" is used for flow variables, i.e. counting new events (infections/deaths/recoveries) on each timestep
        The prefix "n" is used for stock variables, i.e. counting the total number in any given state (sus/inf/rec/etc) on any particular timestep
        The prefix "cum\_" is used for cumulative variables, i.e. counting the total number that have ever been in a given state at some point in the sim
        Note that, by definition, n_dead is the same as cum_deaths and n_recovered is the same as cum_recoveries, so we only define the cumulative versions
        '''

        def init_res(*args, **kwargs):
            ''' Initialize a single result object '''
            output = cvb.Result(*args, **kwargs, npts=self.npts)
            return output

        dcols = cvd.default_colors # Shorten default colors

        # Stock variables
        for key,label in cvd.result_stocks.items():
            self.results[f'n_{key}'] = init_res(label, color=dcols[key])
        self.results['n_susceptible'].scale = 'static'
        self.results['bed_capacity']  = init_res('Bed demand relative to capacity', scale=False)

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


    def load_population(self, popfile=None, **kwargs):
        '''
        Load the population dictionary from file -- typically done automatically
        as part of sim.initialize(load_pop=True).

        Args:
            popfile (str): name of the file to load
        '''
        if popfile is None and self.popfile is not None:
            popfile = self.popfile
        if popfile is not None:
            filepath = sc.makefilepath(filename=popfile, **kwargs)
            self.popdict = sc.loadobj(filepath)
            n_actual = len(self.popdict['uid'])
            n_expected = self['pop_size']
            if n_actual != n_expected:
                errormsg = f'Wrong number of people ({n_expected:n} requested, {n_actual:n} actual) -- please change "pop_size" to match or regenerate the file'
                raise ValueError(errormsg)
            if self['verbose']:
                print(f'Loaded population from {filepath}')
        return


    def init_people(self, save_pop=False, load_pop=False, popfile=None, verbose=None, **kwargs):
        '''
        Create the people.

        Args:
            save_pop (bool): if true, save the population to popfile
            load_pop (bool): if true, load the population from popfile
            popfile (str): filename to load/save the population
            verbose (int): detail to prnt
        '''

        # Handle inputs
        if verbose is None:
            verbose = self['verbose']
        if verbose:
            print(f'Initializing sim with {self["pop_size"]:0n} people for {self["n_days"]} days')
        if load_pop and self.popdict is None:
            self.load_population(popfile=popfile)

        # Actually make the people
        cvpop.make_people(self, save_pop=save_pop, popfile=popfile, verbose=verbose, **kwargs)
        self.people.initialize()

        # Create the seed infections
        inds = np.arange(int(self['pop_infected']))
        self.people.infect(inds=inds)
        for ind in inds:
            self.people.transtree.linelist[ind] = dict(source=None, target=ind, date=self.t, layer='seed_infection')

        return


    def step(self):
        '''
        Step the simulation forward in time
        '''

        # Set the time and if we have reached the end of the simulation, then do nothing
        t = self.t
        if t >= self.npts:
            return

        # Perform initial operations
        self.rescale() # Check if we need to rescale
        people   = self.people # Shorten this for later use
        flows    = people.update_states_pre(t=t) # Update the state of everyone and count the flows
        contacts = people.update_contacts() # Compute new contacts
        bed_max  = people.count('severe') > self['n_beds'] # Check for a bed constraint

        # Randomly infect some people (imported infections)
        n_imports = cvu.poisson(self['n_imports']) # Imported cases
        if n_imports>0:
            imporation_inds = cvu.choose(max_n=len(people), n=n_imports)
            flows['new_infections'] += people.infect(inds=imporation_inds, bed_max=bed_max)
            for ind in imporation_inds:
                self.people.transtree.linelist[ind] = dict(source=None, target=ind, date=self.t, layer='importation')

        # Apply interventions
        for intervention in self['interventions']:
            intervention.apply(self)
        if self['interv_func'] is not None: # Apply custom intervention function
            self =self['interv_func'](self)

        flows = people.update_states_post(flows) # Check for state changes after interventions

        # Compute the probability of transmission
        beta         = cvd.default_float(self['beta'])
        asymp_factor = cvd.default_float(self['asymp_factor'])
        diag_factor  = cvd.default_float(self['diag_factor'])
        frac_time    = cvd.default_float(self['viral_dist']['frac_time'])
        load_ratio   = cvd.default_float(self['viral_dist']['load_ratio'])
        date_inf     = people.date_infectious
        date_rec     = people.date_recovered
        date_dead    = people.date_dead
        viral_load = cvu.compute_viral_load(t, date_inf, date_rec, date_dead, frac_time, load_ratio)

        for lkey,layer in contacts.items():
            sources = layer['p1']
            targets = layer['p2']
            betas   = layer['beta']

            # Compute relative transmission and susceptibility
            rel_trans  = people.rel_trans
            rel_sus    = people.rel_sus
            symp       = people.symptomatic
            diag       = people.diagnosed
            quar       = people.quarantined
            quar_eff   = cvd.default_float(self['quar_eff'][lkey])
            beta_layer = cvd.default_float(self['beta_layer'][lkey])
            rel_trans, rel_sus = cvu.compute_trans_sus(rel_trans, rel_sus, beta_layer, viral_load, symp, diag, quar, asymp_factor, diag_factor, quar_eff)

            # Calculate actual transmission
            target_inds, edge_inds = cvu.compute_infections(beta, sources, targets, betas, rel_trans, rel_sus) # Calculate transmission!
            flows['new_infections'] += people.infect(inds=target_inds, bed_max=bed_max) # Actually infect people

            # Store the transmission tree
            for ind in edge_inds:
                source = sources[ind]
                target = targets[ind]
                transdict = dict(source=source, target=target, date=self.t, layer=lkey)
                self.people.transtree.linelist[target] = transdict
                self.people.transtree.targets[source].append(transdict)

        # Update counts for this time step: stocks
        for key in cvd.result_stocks.keys():
            self.results[f'n_{key}'][t] = people.count(key)
        self.results['bed_capacity'][t] = self.results['n_severe'][t]/self['n_beds'] if self['n_beds']>0 else np.nan

        # Update counts for this time step: flows
        for key,count in flows.items():
            self.results[key][t] += count

        # Tidy up
        self.t += 1
        return


    def rescale(self):
        ''' Dynamically rescale the population '''
        if self['rescale']:
            t = self.t
            pop_scale = self['pop_scale']
            current_scale = self.rescale_vec[t]
            if current_scale < pop_scale: # We have room to rescale
                n_not_sus = self.people.count_not('susceptible')
                n_people = len(self.people)
                if n_not_sus / n_people > self['rescale_threshold']: # Check if we've reached point when we want to rescale
                    max_ratio = pop_scale/current_scale # We don't want to exceed this
                    scaling_ratio = min(self['rescale_factor'], max_ratio)
                    self.rescale_vec[t+1:] *= scaling_ratio # Update the rescaling factor from here on
                    n = int(n_people*(1.0-1.0/scaling_ratio)) # For example, rescaling by 2 gives n = 0.5*n_people
                    new_sus_inds = cvu.choose(max_n=n_people, n=n) # Choose who to make susceptible again
                    self.people.make_susceptible(new_sus_inds)
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

        # Initialize
        T = sc.tic()
        if not self.initialized:
            self.initialize()
        if verbose is None:
            verbose = self['verbose']

        # Main simulation loop
        for t in self.tvec:

            # Print progress
            if verbose >= 1:
                elapsed = sc.toc(output=True)
                simlabel = f'"{self.label}": ' if self.label else ''
                string = f'  Running {simlabel}day {t:2.0f}/{self.pars["n_days"]} ({elapsed:0.2f} s) '
                if verbose >= 2:
                    sc.heading(string)
                elif verbose == 1:
                    cvm.progressbar(t+1, self.npts, label=string, newline=True)

            # Do the heavy lifting -- actually run the model!
            self.step()

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
        sc.printv(f'Run finished after {elapsed:0.2f} s.\n', 1, verbose)
        self.summary = self.summary_stats(verbose=verbose)
        if do_plot: # Optionally plot
            self.plot(**kwargs)

        return self.results


    def finalize(self, verbose=None):
        ''' Compute final results, likelihood, etc. '''

        # Scale the results
        for reskey in self.result_keys():
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
        self.results = sc.objdict(self.results)
        self.results_ready = True
        self.initialized = False # To enable re-running

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
        self.results['doubling_time'][:window] = np.nan
        for t in range(window, self.npts):
            infections_now = cum_infections[t]
            infections_prev = cum_infections[t-window]
            r = infections_now/infections_prev
            if r > 1:  # Avoid divide by zero
                doubling_time = window*np.log(2)/np.log(r)
                doubling_time = min(doubling_time, max_doubling_time) # Otherwise, it's unbounded
                self.results['doubling_time'][t] = doubling_time
        return


    def compute_r_eff(self, window=7):
        '''
        Effective reproductive number based on number of people each person infected.

        Args:
            window (int): the size of the window used (larger values are more accurate but less precise)

        '''

        # Initialize arrays to hold sources and targets infected each day
        sources = np.zeros(self.npts)
        targets = np.zeros(self.npts)
        window = int(window)

        for t in self.tvec:

            # Sources are easy -- count up the arrays
            recov_inds   = cvu.true(t == self.people.date_recovered) # Find people who recovered on this timestep
            dead_inds    = cvu.true(t == self.people.date_dead)  # Find people who died on this timestep
            outcome_inds = np.concatenate((recov_inds, dead_inds))
            sources[t]   = len(outcome_inds)

            # Targets are hard -- loop over the transmission tree
            for ind in outcome_inds:
                targets[t] += len(self.people.transtree.targets[ind])

        # Populate the array -- to avoid divide-by-zero, skip indices that are 0
        inds = sc.findinds(sources>0)
        r_eff = np.zeros(self.npts)*np.nan
        r_eff[inds] = targets[inds]/sources[inds]

        # use stored weights calculate the moving average over the window of timesteps, n
        num = np.nancumsum(r_eff * sources)
        num[window:] = num[window:] - num[:-window]
        den = np.cumsum(sources)
        den[window:] = den[window:] - den[:-window]

        # avoid dividing by zero
        values = np.zeros(num.shape)*np.nan
        ind = den > 0
        values[ind] = num[ind]/den[ind]

        self.results['r_eff'].values = values

        return


    def compute_gen_time(self):
        '''
        Calculate the generation time (or serial interval) there are two
        ways to do this calculation. The 'true' interval (exposure time to
        exposure time) or 'clinical' (symptom onset to symptom onset).
        '''

        intervals1 = np.zeros(len(self.people))
        intervals2 = np.zeros(len(self.people))
        pos1 = 0
        pos2 = 0
        targets = self.people.transtree.targets
        date_exposed = self.people.date_exposed
        date_symptomatic = self.people.date_symptomatic
        for p in range(len(self.people)):
            if len(targets[p])>0:
                for target_ind in targets[p]:
                    intervals1[pos1] = date_exposed[target_ind] - date_exposed[p]
                    pos1 += 1
                    if not np.isnan(date_symptomatic[p]):
                        if not np.isnan(date_symptomatic[target_ind]):
                            intervals2[pos2] = date_symptomatic[target_ind] - date_symptomatic[p]
                            pos2 += 1

        self.results['gen_time'] = {
                'true':         np.mean(intervals1[:pos1]),
                'true_std':     np.std(intervals1[:pos1]),
                'clinical':     np.mean(intervals2[:pos2]),
                'clinical_std': np.std(intervals2[:pos2])}
        return


    def likelihood(self, weights=None, verbose=None, eps=1e-16):
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

        for key in set(self.result_keys()).intersection(self.data.columns): # For keys present in both the results and in the data
            weight = weights.get(key, 1) # Use the provided weight if present, otherwise default to 1
            for d, datum in self.data[key].iteritems():
                if np.isfinite(datum):
                    if d in model_dates:
                        estimate = self.results[key][model_dates.index(d)]
                        if np.isfinite(datum) and np.isfinite(estimate):
                            if (datum == 0) and (estimate == 0):
                                p = 1.0
                            else:
                                p = cvm.poisson_test(datum, estimate)
                            p = max(p, eps)
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
        for key in self.result_keys():
            summary[key] = self.results[key][-1]
            if key.startswith('cum_'):
                summary_str += f'   {summary[key]:5.0f} {self.results[key].name.lower()}\n'
        sc.printv(summary_str, 1, verbose)

        return summary


    def plot(self, to_plot=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
             scatter_args=None, axis_args=None, legend_args=None, as_dates=True, dateformat=None,
             interval=None, n_cols=1, font_size=18, font_family=None, use_grid=True, use_commaticks=True,
             log_scale=False, do_show=True, verbose=None):
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
            log_scale (bool or list): Whether or not to plot the y-axis with a log scale; if a list, panels to show as log
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
            if log_scale:
                if isinstance(log_scale, list):
                    if title in log_scale:
                        ax.set_yscale('log')
                else:
                    ax.set_yscale('log')
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

        **Examples**::

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
