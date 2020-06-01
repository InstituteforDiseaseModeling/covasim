'''
Additional analysis functions that are not part of the core Covasim workflow,
but which are useful for particular investigations. Currently, this just consists
of the transmission tree.
'''

import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc
from . import misc as cvm
from . import interventions as cvi


__all__ = ['Analyzer', 'snapshot', 'age_histogram', 'Fit', 'TransTree']


class Analyzer(sc.prettyobj):
    '''
    Base class for analyzers. Based on the Intervention class.

    Args:
        label (str): a label for the intervention (used for ease of identification)
    '''

    def __init__(self, label=None):
        self.label = label # e.g. "Record ages"
        self.initialized = False
        return


    def initialize(self, sim):
        '''
        Initialize the analyzer, e.g. convert date strings to integers.
        '''
        self.initialized = True
        return


    def apply(self, sim):
        '''
        Apply analyzer at each time point. The analyzer has full access to the
        sim object, and typically stores data/results in itself.

        Args:
            sim: the Sim instance
        '''
        raise NotImplementedError


class snapshot(Analyzer):
    '''
    Analyzer that takes a "snapshot" of the sim.people array at specified points
    in time, and saves them to itself. To retrieve them, you can either access
    the dictionary directly, or use the get() method.

    Args:
        days (list): list of ints/strings/date objects, the days on which to take the snapshot
        kwargs (dict): passed to Intervention()


    **Example**::

        sim = cv.Sim(analyzers=cv.snapshot('2020-04-04', '2020-04-14'))
        sim.run()
        snapshot = sim['analyzers'][0]
        people = snapshot.snapshots[0]            # Option 1
        people = snapshot.snapshots['2020-04-04'] # Option 2
        people = snapshot.get('2020-04-14')       # Option 3
        people = snapshot.get(34)                 # Option 4
        people = snapshot.get()                   # Option 5
    '''

    def __init__(self, days, *args, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        days = sc.promotetolist(days) # Combine multiple days
        days.extend(args) # Include additional arguments, if present
        self.days      = days # Converted to integer representations
        self.dates     = None # String representations
        self.start_day = None # Store the start date of the simulation
        self.snapshots = sc.odict() # Store the actual snapshots
        return


    def initialize(self, sim):
        self.start_day = sim['start_day'] # Store the simulation start day
        self.days = cvi.process_days(sim, self.days) # Ensure days are in the right format
        self.dates = [sim.date(day) for day in self.days] # Store as date strings
        self.initialized = True
        return


    def apply(self, sim):
        for ind in cvi.find_day(self.days, sim.t):
            date = self.dates[ind]
            self.snapshots[date] = sc.dcp(sim.people) # Take snapshot!
        return


    def get(self, key=None):
        ''' Retrieve a snapshot from the given key (int, str, or date) '''
        if key is None:
            key = self.days[0]
        day  = cvm.day(key, start_day=self.start_day)
        date = cvm.date(day, start_date=self.start_day, as_date=False)
        if date in self.snapshots:
            snapshot = self.snapshots[date]
        else:
            dates = ', '.join(list(self.snapshots.keys()))
            errormsg = f'Could not find snapshot date {date} (day {day}): choices are {dates}'
            raise sc.KeyNotFoundError(errormsg)
        return snapshot


class age_histogram(Analyzer):
    '''
    Analyzer that takes a "snapshot" of the sim.people array at specified points
    in time, and saves them to itself. To retrieve them, you can either access
    the dictionary directly, or use the get() method. You can also apply this
    analyzer directly to a sim object.

    Args:
        days    (list): list of ints/strings/date objects, the days on which to calculate the histograms (default: last day)
        states  (list): which states of people to record (default: exposed, tested, diagnosed, dead)
        edges   (list): edges of age bins to use (default: 10 year bins from 0 to 100)
        datafile (str): the name of the data file to load in for comparison, or a dataframe of data (optional)
        sim      (Sim): only used if the analyzer is being used after a sim has already been run
        kwargs  (dict): passed to Intervention()

    **Examples**::

        sim = cv.Sim(analyzers=cv.age_histogram())
        sim.run()
        agehist = sim['analyzers'][0].get()

        agehist = cv.age_histogram(sim=sim)
    '''

    def __init__(self, days=None, states=None, edges=None, datafile=None, sim=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.days      = days # To be converted to integer representations
        self.edges     = edges # Edges of age bins
        self.states    = states # States to save
        self.datafile  = datafile # Data file to load
        self.bins      = None # Age bins, calculated from edges
        self.dates     = None # String representations of dates
        self.start_day = None # Store the start date of the simulation
        self.data      = None # Store the loaded data
        self.hists = sc.odict() # Store the actual snapshots
        self.window_hists = None # Store the histograms for individual windows -- populated by compute_windows()
        if sim is not None: # Process a supplied simulation
            self.from_sim(sim)
        return


    def initialize(self, sim):

        # Handle days
        self.start_day = cvm.date(sim['start_day'], as_date=False) # Get the start day, as a string
        self.end_day   = cvm.date(sim['end_day'], as_date=False) # Get the start day, as a string
        if self.days is None:
            self.days = self.end_day # If no day is supplied, use the last day
        self.days = cvi.process_days(sim, self.days) # Ensure days are in the right format
        self.days = np.sort(self.days) # Ensure they're in order
        self.dates = [sim.date(day) for day in self.days] # Store as date strings
        max_hist_day = self.days[-1]
        max_sim_day = sim.day(self.end_day)
        if max_hist_day > max_sim_day:
            errormsg = f'Cannot create histogram for {self.dates[-1]} (day {max_hist_day}) because the simulation ends on {self.end_day} (day {max_sim_day})'
            raise ValueError(errormsg)

        # Handle edges and age bins
        if self.edges is None: # Default age bins
            self.edges = np.linspace(0,100,11)
        self.bins = self.edges[:-1] # Don't include the last edge in the bins

        # Handle states
        if self.states is None:
            self.states = ['exposed', 'dead', 'tested', 'diagnosed']
        self.states = sc.promotetolist(self.states)
        for s,state in enumerate(self.states):
            self.states[s] = state.replace('date_', '') # Allow keys starting with date_ as input, but strip it off here

        # Handle the data file
        if self.datafile is not None:
            if sc.isstring(self.datafile):
                self.data = cvm.load_data(self.datafile, check_date=False)
            else:
                self.data = self.datafile # Use it directly
                self.datafile = None

        self.initialized = True

        return


    def apply(self, sim):
        for ind in cvi.find_day(self.days, sim.t):
            date = self.dates[ind] # Find the date for this index
            self.hists[date] = sc.objdict() # Initialize the dictionary
            scale  = sim.rescale_vec[sim.t] # Determine current scale factor
            age    = sim.people.age # Get the age distribution,since used heavily
            self.hists[date]['bins'] = self.bins # Copy here for convenience
            for state in self.states: # Loop over each state
                inds = sim.people.defined(f'date_{state}') # Pull out people for which this state is defined
                self.hists[date][state] = np.histogram(age[inds], bins=self.edges)[0]*scale # Actually count the people
        return


    def get(self, key=None):
        ''' Retrieve a specific histogram from the given key (int, str, or date) '''
        if key is None:
            key = self.days[0]
        day  = cvm.day(key, start_day=self.start_day)
        date = cvm.date(day, start_date=self.start_day, as_date=False)
        if date in self.hists:
            hists = self.hists[date]
        else:
            dates = ', '.join(list(self.hists.keys()))
            errormsg = f'Could not find histogram date {date} (day {day}): choices are {dates}'
            raise sc.KeyNotFoundError(errormsg)
        return hists


    def compute_windows(self):
        ''' Convert cumulative histograms to windows '''
        if len(self.hists)<2:
            errormsg = f'You must have at least two dates specified to compute a window'
            raise ValueError(errormsg)

        self.window_hists = sc.objdict()
        for d,end_date,hists in self.hists.enumitems():
            if d==0: # Copy the first one
                start_date = self.start_day
                self.window_hists[f'{start_date} to {end_date}'] = self.hists[end_date]
            else:
                start_date = self.dates[d-1]
                datekey = f'{start_date} to {end_date}'
                self.window_hists[datekey] = sc.objdict() # Initialize the dictionary
                self.window_hists[datekey]['bins'] = self.hists[end_date]['bins']
                for state in self.states: # Loop over each state
                    self.window_hists[datekey][state] = self.hists[end_date][state] - self.hists[start_date][state]

        return


    def from_sim(self, sim):
        ''' Create an age histogram from an already run sim '''
        self.initialize(sim)
        self.apply(sim)
        return


    def plot(self, windows=False, width=0.8, color='#F8A493', font_size=18, fig_args=None, axis_args=None, data_args=None):
        '''
        Simple method for plotting the histograms.

        Args:
            windows (bool): whether to plot windows instead of cumulative counts
            width (float): width of bars
            color (hex or rgb): the color of the bars
            font_size (float): size of font
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            data_args (dict): 'width', 'color', and 'offset' arguments for the data
        '''

        # Handle inputs
        fig_args = sc.mergedicts(dict(figsize=(24,15)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.08, right=0.92, bottom=0.08, top=0.92), axis_args)
        d_args = sc.objdict(sc.mergedicts(dict(width=0.3, color='#000000', offset=0), data_args))
        pl.rcParams['font.size'] = font_size

        # Initialize
        n_plots = len(self.states)
        n_rows = np.ceil(np.sqrt(n_plots)) # Number of subplot rows to have
        n_cols = np.ceil(n_plots/n_rows) # Number of subplot columns to have
        figs = []

        # Handle windows and what to plot
        if windows:
            if self.window_hists is None:
                self.compute_windows()
            histsdict = self.window_hists
        else:
            histsdict = self.hists
        if not len(histsdict):
            errormsg = f'Cannot plot since no histograms were recorded (schuled days: {self.days})'
            raise ValueError(errormsg)

        # Make the figure(s)
        for date,hists in histsdict.items():
            figs += [pl.figure(**fig_args)]
            pl.subplots_adjust(**axis_args)
            bins = hists['bins']
            barwidth = width*(bins[1] - bins[0]) # Assume uniform width
            for s,state in enumerate(self.states):
                pl.subplot(n_rows, n_cols, s+1)
                pl.bar(bins, hists[state], width=barwidth, facecolor=color, label=f'Number {state}')
                if self.data and state in self.data:
                    data = self.data[state]
                    pl.bar(bins+d_args.offset, data, width=barwidth*d_args.width, facecolor=d_args.color, label='Data')
                pl.xlabel('Age')
                pl.ylabel('Count')
                pl.xticks(ticks=bins)
                pl.legend()
                preposition = 'from' if windows else 'by'
                pl.title(f'Number of people {state} {preposition} {date}')

        return figs


class Fit(sc.prettyobj):
    '''
    A class for calculating the fit between the model and the data. Note the
    following terminology is used here:

        - fit: nonspecific term for how well the model matches the data
        - difference: the absolute numerical differences between the model and the data (one time series per result)
        - goodness-of-fit: the result of passing the difference through a statistical function, such as mean squared error
        - loss: the goodness-of-fit for each result multiplied by user-specified weights (one time series per result)
        - mismatch: the sum of all the loses (a single scalar value) -- this is the value to be minimized during calibration

    Args:
        sim (Sim): the sim object
        weights (dict): the relative weight to place on each result
        keys (list): the keys to use in the calculation
        method (str): the method to be used to calculate the goodness-of-fit
        custom (dict): a custom dictionary of additional data to fit; format is e.g. {'<label>':{'data':[1,2,3], 'sim':[1,2,4], 'weights':2.0}}
        compute (bool): whether to compute the mismatch immediately
        verbose (bool): detail to print

    **Example**::

        sim = cv.Sim()
        sim.run()
        fit = sim.compute_fit()
        fit.plot()
    '''

    def __init__(self, sim, weights=None, keys=None, method=None, custom=None, compute=True, verbose=False):

        # Handle inputs
        self.weights = weights
        self.custom  = sc.mergedicts(custom)
        self.verbose = verbose
        self.weights = sc.mergedicts({'cum_deaths':10, 'cum_diagnoses':5}, weights)
        self.keys    = keys

        # Copy data
        if sim.data is None:
            errormsg = 'Model fit cannot be calculated until data are loaded'
            raise RuntimeError(errormsg)
        self.data = sim.data

        # Copy sim results
        if not sim.results_ready:
            errormsg = 'Model fit cannot be calculated until results are run'
            raise RuntimeError(errormsg)
        self.sim_results = sc.objdict()
        for key in sim.result_keys() + ['t', 'date']:
            self.sim_results[key] = sim.results[key]
        self.sim_npts = sim.npts # Number of time points in the sim

        # Copy other things
        self.sim_dates = sim.datevec.tolist()

        # These are populated during initialization
        self.inds         = sc.objdict() # To store matching indices between the data and the simulation
        self.inds.sim     = sc.objdict() # For storing matching indices in the sim
        self.inds.data    = sc.objdict() # For storing matching indices in the data
        self.date_matches = sc.objdict() # For storing matching dates, largely for plotting
        self.pair         = sc.objdict() # For storing perfectly paired points between the data and the sim
        self.diffs        = sc.objdict() # Differences between pairs
        self.gofs         = sc.objdict() # Goodness-of-fit for differences
        self.losses       = sc.objdict() # Weighted goodness-of-fit
        self.mismatches   = sc.objdict() # Final mismatch values
        self.mismatch     = None # The final value

        if compute:
            self.compute()

        return


    def compute(self):
        ''' Perform all required computations '''
        self.reconcile_inputs() # Find matching values
        self.compute_diffs() # Perform calculations
        self.compute_gofs()
        self.compute_losses()
        self.compute_mismatch()
        return self.mismatch


    def reconcile_inputs(self):
        ''' Find matching keys and indices between the model and the data '''

        data_cols = self.data.columns
        if self.keys is None:
            sim_keys = self.sim_results.keys()
            intersection = list(set(sim_keys).intersection(data_cols)) # Find keys in both the sim and data
            self.keys = [key for key in sim_keys if key in intersection and key.startswith('cum_')] # Only keep cumulative keys
            if not len(self.keys):
                errormsg = f'No matches found between simulation result keys ({sim_keys}) and data columns ({data_cols})'
                raise sc.KeyNotFoundError(errormsg)
        mismatches = [key for key in self.keys if key not in data_cols]
        if len(mismatches):
            mismatchstr = ', '.join(mismatches)
            errormsg = f'The following requested key(s) were not found in the data: {mismatchstr}'
            raise sc.KeyNotFoundError(errormsg)

        for key in self.keys: # For keys present in both the results and in the data
            self.inds.sim[key]  = []
            self.inds.data[key] = []
            self.date_matches[key] = []
            count = -1
            for d, datum in self.data[key].iteritems():
                count += 1
                if np.isfinite(datum):
                    if d in self.sim_dates:
                        self.date_matches[key].append(d)
                        self.inds.sim[key].append(self.sim_dates.index(d))
                        self.inds.data[key].append(count)
            self.inds.sim[key]  = np.array(self.inds.sim[key])
            self.inds.data[key] = np.array(self.inds.data[key])

        # Convert into paired points
        for key in self.keys:
            self.pair[key] = sc.objdict()
            sim_inds = self.inds.sim[key]
            data_inds = self.inds.data[key]
            n_inds = len(sim_inds)
            self.pair[key].sim  = np.zeros(n_inds)
            self.pair[key].data = np.zeros(n_inds)
            for i in range(n_inds):
                self.pair[key].sim[i]  = self.sim_results[key].values[sim_inds[i]]
                self.pair[key].data[i] = self.data[key].values[data_inds[i]]

        # Process custom inputs
        self.custom_keys = list(self.custom.keys())
        for key in self.custom.keys():

            # Initialize and do error checking
            custom = self.custom[key]
            c_keys = list(custom.keys())
            if 'sim' not in c_keys or 'data' not in c_keys:
                errormsg = f'Custom input must have "sim" and "data" keys, not {c_keys}'
                raise sc.KeyNotFoundError(errormsg)
            c_data = custom['data']
            c_sim  = custom['sim']
            try:
                assert len(c_data) == len(c_sim)
            except:
                errormsg = f'Custom data and sim must be arrays, and be of the same length: data = {c_data}, sim = {c_sim} could not be processed'
                raise ValueError(errormsg)
            if key in self.pair:
                errormsg = f'You cannot use a custom key "{key}" that matches one of the existing keys: {self.pair.keys()}'
                raise ValueError(errormsg)

            # If all tests pass, simply copy the data
            self.pair[key] = sc.objdict()
            self.pair[key].sim  = c_sim
            self.pair[key].data = c_data

            # Process weight, if available
            wt = custom.get('weight', 1.0) # Attempt to retrieve key 'weight', or use the default if not provided
            wt = custom.get('weights', wt) # ...but also try "weights"
            self.weights[key] = wt # Set the weight

        return


    def compute_diffs(self, absolute=False):
        ''' Find the differences between the sim and the data '''
        for key in self.pair.keys():
            self.diffs[key] = self.pair[key].sim - self.pair[key].data
            if absolute:
                self.diffs[key] = np.abs(self.diffs[key])
        return


    def compute_gofs(self, **kwargs):
        ''' Compute the goodness-of-fit '''
        for key in self.pair.keys():
            actual    = sc.dcp(self.pair[key].data)
            predicted = sc.dcp(self.pair[key].sim)
            self.gofs[key] = cvm.compute_gof(actual, predicted, **kwargs)
        return


    def compute_losses(self):
        ''' Compute the weighted goodness-of-fit '''
        for key in self.gofs.keys():
            if key in self.weights:
                weight = self.weights[key]
                if sc.isiterable(weight): # It's an array
                    len_wt = len(weight)
                    len_sim = self.sim_npts
                    len_match = len(self.gofs[key])
                    if len_wt == len_match: # If the weight already is the right length, do nothing
                        pass
                    elif len_wt == len_sim: # Most typical case: it's the length of the simulation, must trim
                        weight = weight[self.inds.sim[key]] # Trim to matching indices
                    else:
                        errormsg = f'Could not map weight array of length {len_wt} onto simulation of length {len_sim} or data-model matches of length {len_match}'
                        raise ValueError(errormsg)
            else:
                weight = 1.0
            self.losses[key] = self.gofs[key]*weight
        return


    def compute_mismatch(self, use_median=False):
        ''' Compute the final mismatch '''
        for key in self.losses.keys():
            if use_median:
                self.mismatches[key] = np.median(self.losses[key])
            else:
                self.mismatches[key] = np.sum(self.losses[key])
        self.mismatch = self.mismatches[:].sum()
        return self.mismatch


    def plot(self, keys=None, width=0.8, font_size=18, fig_args=None, axis_args=None, plot_args=None):
        '''
        Plot the fit of the model to the data. For each result, plot the data
        and the model; the difference; and the loss (weighted difference). Also
        plots the loss as a function of time.

        Args:
            keys (list): which keys to plot (default, all)
            width (float): bar width
            font_size (float): size of font
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            plot_args (dict): passed to pl.plot()
        '''

        fig_args  = sc.mergedicts(dict(figsize=(36,22)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.3), axis_args)
        plot_args = sc.mergedicts(dict(lw=4, alpha=0.5, marker='o'), plot_args)
        pl.rcParams['font.size'] = font_size

        if keys is None:
            keys = self.keys + self.custom_keys
        n_keys = len(keys)

        loss_ax = None
        colors = sc.gridcolors(n_keys)
        n_rows = 4

        figs = [pl.figure(**fig_args)]
        pl.subplots_adjust(**axis_args)
        main_ax1 = pl.subplot(n_rows, 2, 1)
        main_ax2 = pl.subplot(n_rows, 2, 2)
        bottom = sc.objdict() # Keep track of the bottoms for plotting cumulative
        bottom.a = np.zeros(self.losses[0].shape)
        bottom.b = np.zeros(self.losses[0].shape)
        for k,key in enumerate(keys):
            if key in self.keys: # It's a time series, plot with days and dates
                days      = self.inds.sim[key] # The "days" axis (or not, for custom keys)
                daylabel  = 'Day'
            else: #It's custom, we don't know what it is
                days      = np.arange(len(self.losses[key])) # Just use indices
                daylabel  = 'Index'

            # Cumulative totals can't mix daily and non-daily inputs, so skip custom keys
            if key in self.keys:
                for i,ax in enumerate([main_ax1, main_ax2]):

                    if i == 0:
                        data = self.losses[key]
                        ylabel = 'Daily mismatch'
                        title = f'Daily total mismatch'
                    else:
                        data = np.cumsum(self.losses[key])
                        ylabel = 'Cumulative mismatch'
                        title = f'Cumulative mismatch: {self.mismatch:0.3f}'

                    dates = self.sim_results['date'][days] # Show these with dates, rather than days, as a reference point
                    ax.bar(dates, data, width=width, bottom=bottom[i], color=colors[k], label=f'{key}')

                    if i == 0:
                        bottom[i] += self.losses[key]
                    else:
                        bottom[i] += np.cumsum(self.losses[key])

                    if k == len(self.keys)-1:
                        ax.set_xlabel('Date')
                        ax.set_ylabel(ylabel)
                        ax.set_title(title)
                        ax.legend()

            pl.subplot(n_rows, n_keys, k+1*n_keys+1)
            pl.plot(days, self.pair[key].data, c='k', label='Data', **plot_args)
            pl.plot(days, self.pair[key].sim, c=colors[k], label='Simulation', **plot_args)
            pl.title(key)
            if k == 0:
                pl.ylabel('Time series (counts)')
                pl.legend()

            pl.subplot(n_rows, n_keys, k+2*n_keys+1)
            pl.bar(days, self.diffs[key], width=width, color=colors[k], label='Difference')
            pl.axhline(0, c='k')
            if k == 0:
                pl.ylabel('Differences (counts)')
                pl.legend()

            loss_ax = pl.subplot(n_rows, n_keys, k+3*n_keys+1, sharey=loss_ax)
            pl.bar(days, self.losses[key], width=width, color=colors[k], label='Losses')
            pl.xlabel(daylabel)
            pl.title(f'Total loss: {self.losses[key].sum():0.3f}')
            if k == 0:
                pl.ylabel('Losses')
                pl.legend()

        return figs


class TransTree(sc.prettyobj):
    '''
    A class for holding a transmission tree. There are several different representations
    of the transmission tree: "infection_log" is copied from the people object and is the
    simplest representation. "detailed h" includes additional attributes about the source
    and target. If NetworkX is installed (required for most methods), "graph" includes an
    NX representation of the transmission tree.

    Args:
        sim (Sim): the sim object
        to_networkx (bool): whether to convert the graph to a NetworkX object
    '''

    def __init__(self, sim, to_networkx=False):

        # Pull out each of the attributes relevant to transmission
        attrs = {'age', 'date_exposed', 'date_symptomatic', 'date_tested', 'date_diagnosed', 'date_quarantined', 'date_severe', 'date_critical', 'date_known_contact', 'date_recovered'}

        # Pull out the people and some of the sim results
        people = sim.people
        self.sim_start = sim['start_day'] # Used for filtering later
        self.sim_results = {}
        self.sim_results['t'] = sim.results['t']
        self.sim_results['cum_infections'] = sim.results['cum_infections'].values
        self.n_days = people.t  # people.t should be set to the last simulation timestep in the output (since the Transtree is constructed after the people have been stepped forward in time)
        self.pop_size = len(people)

        # Include the basic line list
        self.infection_log = sc.dcp(people.infection_log)

        # Parse into sources and targets
        self.sources = [None for i in range(self.pop_size)]
        self.targets = [[]   for i in range(self.pop_size)]
        self.source_dates = [None for i in range(self.pop_size)]
        self.target_dates = [[]   for i in range(self.pop_size)]

        for entry in self.infection_log:
            source = entry['source']
            target = entry['target']
            date   = entry['date']
            if source:
                self.sources[target] = source # Each target has at most one source
                self.targets[source].append(target) # Each source can have multiple targets
                self.source_dates[target] = date # Each target has at most one source
                self.target_dates[source].append(date) # Each source can have multiple targets

        # Count the number of targets each person has
        self.n_targets = self.count_targets()

        # Include the detailed transmission tree as well
        self.detailed = self.make_detailed(people)

        # Optionally convert to NetworkX -- must be done on import since the people object is not kept
        if to_networkx:

            # Initialization
            import networkx as nx
            self.graph = nx.DiGraph()

            # Add the nodes
            for i in range(len(people)):
                d = {}
                for attr in attrs:
                    d[attr] = people[attr][i]
                self.graph.add_node(i, **d)

            # Next, add edges from linelist
            for edge in people.infection_log:
                self.graph.add_edge(edge['source'],edge['target'],date=edge['date'],layer=edge['layer'])

        return


    def __len__(self):
        '''
        The length of the transmission tree is the length of the line list,
        which should equal the number of infections.
        '''
        try:
            return len(self.infection_log)
        except:
            return 0


    @property
    def transmissions(self):
        """
        Iterable over edges corresponding to transmission events

        This excludes edges corresponding to seeded infections without a source
        """
        output = []
        for d in self.infection_log:
            if d['source'] is not None:
                output.append([d['source'], d['target']])
        return output


    def day(self, day=None, which=None):
        ''' Convenience function for converting an input to an integer day '''
        if day is not None:
            day = cvm.day(day, start_day=self.sim_start)
        elif which == 'start':
            day = 0
        elif which == 'end':
            day = self.n_days
        return day


    def count_targets(self, start_day=None, end_day=None):
        '''
        Count the number of targets each infected person has. If start and/or end
        days are given, it will only count the targets of people who got infected
        between those dates (it does not, however, filter on the date the target
        got infected).

        Args:
            start_day (int/str): the day on which to start counting people who got infected
            end_day (int/str): the day on which to stop counting people who got infected
        '''

        # Handle start and end days
        start_day = self.day(start_day, which='start')
        end_day   = self.day(end_day,   which='end')

        n_targets = np.nan+np.zeros(self.pop_size)
        for i in range(self.pop_size):
            if self.sources[i] is not None:
                if self.source_dates[i] >= start_day and self.source_dates[i] <= end_day:
                    n_targets[i] = len(self.targets[i])
        n_target_inds = sc.findinds(~np.isnan(n_targets))
        n_targets = n_targets[n_target_inds]
        return n_targets


    def make_detailed(self, people, reset=False):
        ''' Construct a detailed transmission tree, with additional information for each person '''

        detailed = [None]*self.pop_size

        for transdict in self.infection_log:

            # Pull out key quantities
            ddict  = sc.dcp(transdict) # For "detailed dictionary"
            source = ddict['source']
            target = ddict['target']
            ddict['s'] = {} # Source properties
            ddict['t'] = {} # Target properties

            # If the source is available (e.g. not a seed infection), loop over both it and the target
            if source is not None:
                stdict = {'s':source, 't':target}
            else:
                stdict = {'t':target}

            # Pull out each of the attributes relevant to transmission
            attrs = ['age', 'date_symptomatic', 'date_tested', 'date_diagnosed', 'date_quarantined', 'date_severe', 'date_critical', 'date_known_contact']
            for st,stind in stdict.items():
                for attr in attrs:
                    ddict[st][attr] = people[attr][stind]
            if source is not None:
                for attr in attrs:
                    if attr.startswith('date_'):
                        is_attr = attr.replace('date_', 'is_') # Convert date to a boolean, e.g. date_diagnosed -> is_diagnosed
                        ddict['s'][is_attr] = ddict['s'][attr] <= ddict['date'] # These don't make sense for people just infected (targets), only sources

                ddict['s']['is_asymp']   = np.isnan(people.date_symptomatic[source])
                ddict['s']['is_presymp'] = ~ddict['s']['is_asymp'] and ~ddict['s']['is_symptomatic'] # Not asymptomatic and not currently symptomatic
            ddict['t']['is_quarantined'] = ddict['t']['date_quarantined'] <= ddict['date'] # This is the only target date that it makes sense to define since it can happen before infection

            detailed[target] = ddict

        return detailed


    def r0(self, recovered_only=False):
        """
        Return average number of transmissions per person

        This doesn't include seed transmissions. By default, it also doesn't adjust
        for length of infection (e.g. people infected towards the end of the simulation
        will have fewer transmissions because their infection may extend past the end
        of the simulation, these people are not included). If 'recovered_only=True'
        then the downstream transmissions will only be included for people that recover
        before the end of the simulation, thus ensuring they all had the same amount of
        time to transmit.
        """
        n_infected = []
        try:
            for i, node in self.graph.nodes.items():
                if i is None or np.isnan(node['date_exposed']) or (recovered_only and node['date_recovered']>self.n_days):
                    continue
                n_infected.append(self.graph.out_degree(i))
        except Exception as E:
            errormsg = f'Unable to compute r0 ({str(E)}): you may need to reinitialize the transmission tree with to_networkx=True'
            raise RuntimeError(errormsg)
        return np.mean(n_infected)


    def plot(self, *args, **kwargs):
        ''' Plot the transmission tree '''

        fig_args = kwargs.get('fig_args', dict(figsize=(16, 10)))

        ttlist = []
        for source_ind, target_ind in self.transmissions:
            ddict = self.detailed[target_ind]
            source = ddict['s']
            target = ddict['t']

            tdict = {}
            tdict['date'] =  ddict['date']
            tdict['layer'] =  ddict['layer']
            tdict['s_asymp'] =  np.isnan(source['date_symptomatic']) # True if they *never* became symptomatic
            tdict['s_presymp'] =  ~tdict['s_asymp'] and tdict['date']<source['date_symptomatic'] # True if they became symptomatic after the transmission date
            tdict['s_sev'] = source['date_severe'] < tdict['date']
            tdict['s_crit'] = source['date_critical'] < tdict['date']
            tdict['s_diag'] = source['date_diagnosed'] < tdict['date']
            tdict['s_quar'] = source['date_quarantined'] < tdict['date']
            tdict['t_quar'] = target['date_quarantined'] < tdict['date'] # What if the target was released from quarantine?
            ttlist.append(tdict)

        df = pd.DataFrame(ttlist).rename(columns={'date': 'Day'})
        df = df.loc[df['layer'] != 'seed_infection']

        df['Stage'] = 'Symptomatic'
        df.loc[df['s_asymp'], 'Stage'] = 'Asymptomatic'
        df.loc[df['s_presymp'], 'Stage'] = 'Presymptomatic'

        df['Severity'] = 'Mild'
        df.loc[df['s_sev'], 'Severity'] = 'Severe'
        df.loc[df['s_crit'], 'Severity'] = 'Critical'

        fig = pl.figure(**fig_args)
        i = 1;
        r = 2;
        c = 3

        def plot_quantity(key, title, i):
            dat = df.groupby(['Day', key]).size().unstack(key)
            ax = pl.subplot(r, c, i);
            dat.plot(ax=ax, legend=None)
            pl.legend(title=None)
            ax.set_title(title)

        to_plot = {
            'layer': 'Layer',
            'Stage': 'Source stage',
            's_diag': 'Source diagnosed',
            's_quar': 'Source quarantined',
            't_quar': 'Target quarantined',
            'Severity': 'Symptomatic source severity'
        }
        for i, (key, title) in enumerate(to_plot.items()):
            plot_quantity(key, title, i + 1)

        return fig


    def animate(self, *args, **kwargs):
        '''
        Animate the transmission tree.

        Args:
            animate    (bool):  whether to animate the plot (otherwise, show when finished)
            verbose    (bool):  print out progress of each frame
            markersize (int):   size of the markers
            sus_color  (list):  color for susceptibles
            fig_args   (dict):  arguments passed to pl.figure()
            axis_args  (dict):  arguments passed to pl.subplots_adjust()
            plot_args  (dict):  arguments passed to pl.plot()
            delay      (float): delay between frames in seconds
            font_size  (int):   size of the font
            colors     (list):  color of each person
            cmap       (str):   colormap for each person (if colors is not supplied)

        Returns:
            fig: the figure object
        '''

        # Settings
        animate = kwargs.get('animate', True)
        verbose = kwargs.get('verbose', False)
        msize = kwargs.get('markersize', 10)
        sus_color = kwargs.get('sus_color', [0.5, 0.5, 0.5])
        fig_args = kwargs.get('fig_args', dict(figsize=(24, 16)))
        axis_args = kwargs.get('axis_args', dict(left=0.10, bottom=0.05, right=0.85, top=0.97, wspace=0.25, hspace=0.25))
        plot_args = kwargs.get('plot_args', dict(lw=2, alpha=0.5))
        delay = kwargs.get('delay', 0.2)
        font_size = kwargs.get('font_size', 18)
        colors = kwargs.get('colors', None)
        cmap = kwargs.get('cmap', 'parula')
        pl.rcParams['font.size'] = font_size
        if colors is None:
            colors = sc.vectocolor(self.pop_size, cmap=cmap)

        # Initialization
        n = self.n_days + 1
        frames = [list() for i in range(n)]
        tests = [list() for i in range(n)]
        diags = [list() for i in range(n)]
        quars = [list() for i in range(n)]

        # Construct each frame of the animation
        for ddict in self.detailed:  # Loop over every person
            if ddict is None:
                continue # Skip the 'None' node corresponding to seeded infections

            frame = {}
            tdq = {}  # Short for "tested, diagnosed, or quarantined"
            target = ddict['t']
            target_ind = ddict['target']

            if not np.isnan(ddict['date']): # If this person was infected

                source_ind = ddict['source'] # Index of the person who infected the target

                target_date = ddict['date']
                if source_ind is not None:  # Seed infections and importations won't have a source
                    source_date = self.detailed[source_ind]['date']
                else:
                    source_ind = 0
                    source_date = 0

                # Construct this frame
                frame['x'] = [source_date, target_date]
                frame['y'] = [source_ind, target_ind]
                frame['c'] = colors[source_ind]
                frame['i'] = True  # If this person is infected
                frames[int(target_date)].append(frame)

                # Handle testing, diagnosis, and quarantine
                tdq['t'] = target_ind
                tdq['d'] = target_date
                tdq['c'] = colors[int(target_ind)]
                date_t = target['date_tested']
                date_d = target['date_diagnosed']
                date_q = target['date_known_contact']
                if ~np.isnan(date_t) and date_t < n:
                    tests[int(date_t)].append(tdq)
                if ~np.isnan(date_d) and date_d < n:
                    diags[int(date_d)].append(tdq)
                if ~np.isnan(date_q) and date_q < n:
                    quars[int(date_q)].append(tdq)

            else:
                frame['x'] = [0]
                frame['y'] = [target_ind]
                frame['c'] = sus_color
                frame['i'] = False
                frames[0].append(frame)

        # Configure plotting
        fig = pl.figure(**fig_args)
        pl.subplots_adjust(**axis_args)
        ax = fig.add_subplot(1, 1, 1)

        # Create the legend
        ax2 = pl.axes([0.85, 0.05, 0.14, 0.9])
        ax2.axis('off')
        lcol = colors[0]
        na = np.nan  # Shorten
        pl.plot(na, na, '-', c=lcol, **plot_args, label='Transmission')
        pl.plot(na, na, 'o', c=lcol, markersize=msize, **plot_args, label='Source')
        pl.plot(na, na, '*', c=lcol, markersize=msize, **plot_args, label='Target')
        pl.plot(na, na, 'o', c=lcol, markersize=msize * 2, fillstyle='none', **plot_args, label='Tested')
        pl.plot(na, na, 's', c=lcol, markersize=msize * 1.2, **plot_args, label='Diagnosed')
        pl.plot(na, na, 'x', c=lcol, markersize=msize * 2.0, label='Known contact')
        pl.legend()

        # Plot the animation
        pl.sca(ax)
        for day in range(n):
            pl.title(f'Day: {day}')
            pl.xlim([0, n])
            pl.ylim([0, len(self)])
            pl.xlabel('Day')
            pl.ylabel('Person')
            flist = frames[day]
            tlist = tests[day]
            dlist = diags[day]
            qlist = quars[day]
            t_d = tdq['d']
            t_t = tdq['t']
            t_c = tdq['c']
            for f in flist:
                if verbose: print(f)
                x = f['x']
                y = f['y']
                c = f['c']
                pl.plot(x[0], y[0], 'o', c=c, markersize=msize, **plot_args)  # Plot sources
                pl.plot(x, y, '-', c=c, **plot_args)  # Plot transmission lines
                if f['i']:  # If this person is infected
                    pl.plot(x[1], y[1], '*', c=c, markersize=msize, **plot_args)  # Plot targets
            for tdq in tlist: pl.plot(t_d, t_t, 'o', c=t_c, markersize=msize * 2, fillstyle='none')  # Tested; No alpha for this
            for tdq in dlist: pl.plot(t_d, t_t, 's', c=t_c, markersize=msize * 1.2, **plot_args)  # Diagnosed
            for tdq in qlist: pl.plot(t_d, t_t, 'x', c=t_c, markersize=msize * 2.0)  # Quarantine; no alpha for this
            pl.plot([0, day], [0.5, 0.5], c='k', lw=5)  # Plot the endless march of time
            if animate:  # Whether to animate
                pl.pause(delay)

        return fig


    def plot_histograms(self, start_day=None, end_day=None, bins=None, width=0.8, fig_args=None, font_size=18):
        '''
        Plots a histogram of the number of transmissions.

        Args:
            start_day (int/str): the day on which to start counting people who got infected
            end_day (int/str): the day on which to stop counting people who got infected
            bins (list): bin edges to use for the histogram
            width (float): width of bars
            fig_args (dict): passed to pl.figure()
            font_size (float): size of font
        '''

        # Process targets
        n_targets = self.count_targets(start_day, end_day)

        # Handle bins
        if bins is None:
            max_infections = n_targets.max()
            bins = np.arange(0, max_infections+2)

        # Analysis
        counts = np.histogram(n_targets, bins)[0]

        bins = bins[:-1] # Remove last bin since it's an edge
        total_counts = counts*bins
        # counts = counts*100/counts.sum()
        # total_counts = total_counts*100/total_counts.sum()
        n_bins = len(bins)
        index = np.linspace(0, 100, len(n_targets))
        sorted_arr = np.sort(n_targets)
        sorted_sum = np.cumsum(sorted_arr)
        sorted_sum = sorted_sum/sorted_sum.max()*100
        change_inds = sc.findinds(np.diff(sorted_arr) != 0)
        max_labels = 15 # Maximum number of ticks and legend entries to plot

        # Plotting
        fig_args = sc.mergedicts(dict(figsize=(24,15)), fig_args)
        pl.rcParams['font.size'] = font_size
        fig = pl.figure(**fig_args)
        pl.set_cmap('Spectral')
        pl.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92)
        colors = sc.vectocolor(n_bins)

        pl.subplot(1,2,1)
        w05 = width*0.5
        w025 = w05*0.5
        pl.bar(bins-w025, counts, width=w05, facecolor='k', label='Number of events')
        for i in range(n_bins):
            label = 'Number of transmissions (events × transmissions per event)' if i==0 else None
            pl.bar(bins[i]+w025, total_counts[i], width=w05, facecolor=colors[i], label=label)
        pl.xlabel('Number of transmissions per person')
        pl.ylabel('Count')
        if n_bins<max_labels:
            pl.xticks(ticks=bins)
        pl.legend()
        pl.title('Numbers of events and transmissions')

        pl.subplot(2,2,2)
        total = 0
        for i in range(n_bins):
            pl.bar(bins[i:], total_counts[i], width=width, bottom=total, facecolor=colors[i])
            total += total_counts[i]
        if n_bins<max_labels:
            pl.xticks(ticks=bins)
        pl.xlabel('Number of transmissions per person')
        pl.ylabel('Number of infections caused')
        pl.title('Number of transmissions, by transmissions per person')

        pl.subplot(2,2,4)
        pl.plot(index, sorted_sum, lw=3, c='k', alpha=0.5)
        n_change_inds = len(change_inds)
        label_inds = np.linspace(0, n_change_inds, max_labels).round() # Don't allow more than this many labels
        for i in range(n_change_inds):
            if i in label_inds: # Don't plot more than this many labels
                label = f'Transmitted to {bins[i+1]:n} people'
            else:
                label = None
            pl.scatter([index[change_inds[i]]], [sorted_sum[change_inds[i]]], s=150, zorder=10, c=[colors[i]], label=label)
        pl.xlabel('Proportion of population, ordered by the number of people they infected (%)')
        pl.ylabel('Proportion of infections caused (%)')
        pl.legend()
        pl.ylim([0, 100])
        pl.grid(True)
        pl.title('Proportion of transmissions, by proportion of population')

        pl.axes([0.30, 0.65, 0.15, 0.2])
        berry      = [0.8, 0.1, 0.2]
        dirty_snow = [0.9, 0.9, 0.9]
        start_day  = self.day(start_day, which='start')
        end_day    = self.day(end_day, which='end')
        pl.axvspan(start_day, end_day, facecolor=dirty_snow)
        pl.plot(self.sim_results['t'], self.sim_results['cum_infections'], lw=2, c=berry)
        pl.xlabel('Day')
        pl.ylabel('Cumulative infections')


        return fig

