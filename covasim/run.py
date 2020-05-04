'''
Functions and classes for running multiple Covasim runs.
'''

#%% Imports
import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc
from collections import defaultdict
from . import misc as cvm
from . import base as cvb
from . import sim as cvs
from . import plotting as cvplt


# Specify all externally visible functions this file defines
__all__ = ['make_metapars', 'MultiSim', 'Scenarios', 'single_run', 'multi_run']



def make_metapars():
    ''' Create default metaparameters for a Scenarios run '''
    metapars = sc.objdict(
        n_runs    = 3, # Number of parallel runs; change to 3 for quick, 11 for real
        noise     = 0.1, # Use noise, optionally
        noisepar  = 'beta',
        rand_seed = 1,
        quantiles = {'low':0.1, 'high':0.9},
        verbose   = 1,
    )
    return metapars


class MultiSim(sc.prettyobj):
    '''
    Class for running multiple copies of a simulation. The parameter n_runs
    controls how many copies of the simulation there will be, if a list of sims
    is not provided.

    Args:
        sims (Sim or list): a single sim or a list of sims
        base_sim (Sim): the sim used for shared properties; if not supplied, the first of the sims provided
        quantiles (dict): the quantiles to use with reduce(), e.g. [0.1, 0.9] or {'low:'0.1, 'high':0.9}
        kwargs (dict): stored in run_args and passed to run()

    Returns:
        msim: a MultiSim object

    **Examples**::

        sim = cv.Sim() # Create the sim
        msim = cv.MultiSim(sim, n_runs=5) # Create the multisim
        msim.run() # Run them in parallel
        msim.combine() # Combine into one sim
        msim.plot() # Plot results

        sim = cv.Sim() # Create the sim
        msim = cv.MultiSim(sim, n_runs=11, noise=0.1) # Set up a multisim with noise
        msim.run() # Run
        msim.reduce() # Compute statistics
        msim.plot() # Plot

        sims = [cv.Sim(beta=0.015*(1+0.02*i)) for i in range(5)] # Create sims
        for sim in sims: sim.run() # Run sims in serial
        msim = cv.MultiSim(sims) # Convert to multisim
        msim.plot() # Plot as single sim
    '''

    def __init__(self, sims=None, base_sim=None, quantiles=None, **kwargs):

        # Handle inputs
        if base_sim is None:
            if isinstance(sims, cvs.Sim):
                base_sim = sims
                sims = None
            elif isinstance(sims, list):
                base_sim = sims[0]
            else:
                errormsg = f'If base_sim is not supplied, sims must be either a sims or a list of sims, not {type(sims)}'
                raise TypeError(errormsg)

        if quantiles is None:
            quantiles = make_metapars()['quantiles']

        self.sims = sims
        self.base_sim = base_sim
        self.quantiles = quantiles
        self.run_args = sc.mergedicts(kwargs)
        self.results = None
        self.which = None # Whether the multisim is to be reduced, combined, etc.

        return


    def __len__(self):
        try:
            return len(self.sims)
        except:
            return 0


    def result_keys(self):
        ''' Attempt to retrieve the results keys from the base sim '''
        try:
            keys = self.base_sim.result_keys()
        except Exception as E:
            errormsg = f'Could not retrieve result keys since base sim not accessible: {str(E)}'
            raise ValueError(errormsg)
        return keys


    def run(self, *args, **kwargs):
        '''
        Run the actual scenarios

        Args:
            debug (bool): if True, runs a single run instead of multiple, which makes debugging easier
            verbose (int): level of detail to print, passed to sim.run()
            kwargs (dict): passed to multi_run() and thence to sim.run()

        Returns:
            None (modifies MultiSim object in place)
        '''
        if self.sims is None:
            sims = self.base_sim
        else:
            sims = self.sims

        kwargs = sc.mergedicts(self.run_args, kwargs)
        self.sims = multi_run(sims, *args, **kwargs)
        return


    def combine(self, output=False):
        ''' Combine multiple sims into a single sim with scaled results '''

        n_runs = len(self)
        combined_sim = sc.dcp(self.sims[0])
        combined_sim.parallelized = {'parallelized':True, 'combined':True, 'n_runs':n_runs}  # Store how this was parallelized
        combined_sim['pop_size'] *= n_runs  # Record the number of people

        for s,sim in enumerate(self.sims[1:]): # Skip the first one
            if combined_sim.people:
                combined_sim.people += sim.people
            for key in sim.result_keys():
                this_res = sim.results[key]
                combined_sim.results[key].values += this_res.values

        # For non-count results (scale=False), rescale them
        for key in combined_sim.result_keys():
            if not combined_sim.results[key].scale:
                combined_sim.results[key].values /= n_runs

        self.base_sim = combined_sim
        self.results = combined_sim.results
        self.which = 'combined'

        if output:
            return self.base_sim
        else:
            return


    def reduce(self, quantiles=None, output=False):
        ''' Combine multiple sims into a single sim with scaled results '''

        if quantiles is None:
            quantiles = self.quantiles
        if not isinstance(quantiles, dict):
            try:
                quantiles = {'low':quantiles[0], 'high':quantiles[1]}
            except Exception as E:
                errormsg = f'Could not figure out how to convert {quantiles} into a quantiles object: must be a dict with keys low, high or a 2-element array ({str(E)})'
                raise ValueError(errormsg)

        # Store information on the sims
        n_runs = len(self)
        reduced_sim = sc.dcp(self.sims[0])
        reduced_sim.parallelized = {'parallelized':True, 'combined':False, 'n_runs':n_runs}  # Store how this was parallelized

        # Perform the statistics
        raw = {}
        reskeys = reduced_sim.result_keys()
        for reskey in reskeys:
            raw[reskey] = np.zeros((reduced_sim.npts, len(self.sims)))
            for s,sim in enumerate(self.sims):
                raw[reskey][:,s] = sim.results[reskey].values

        for reskey in reskeys:
            reduced_sim.results[reskey].values[:] = np.quantile(raw[reskey], q=0.5, axis=1) # Changed from median to mean for smoother plots
            reduced_sim.results[reskey].low       = np.quantile(raw[reskey], q=quantiles['low'],  axis=1)
            reduced_sim.results[reskey].high      = np.quantile(raw[reskey], q=quantiles['high'], axis=1)
        reduced_sim.likelihood() # Recompute the likelihood for the average sim
        reduced_sim.summary_stats(verbose=False) # Recalculate the summary stats

        self.base_sim = reduced_sim
        self.results = reduced_sim.results
        self.which = 'reduced'

        if output:
            return self.base_sim
        else:
            return


    def compare(self, t=-1, sim_inds=None, output=False, do_plot=False, **kwargs):
        '''
        Create a dataframe compare sims at a single point in time.

        Args:
            t (int or str): the day (or date) to do the comparison; default, the end
            sim_inds (list): list of integers of which sims to include (default: all)
            output (bool): whether or not to return the comparison as a dataframe
            do_plot (bool): whether or not to plot the comparison (see also plot_compare())
            kwargs (dict): passed to plot_compare()

        Returns:
            df (dataframe): a dataframe comparison
        '''

        # Handle the indices
        if sim_inds is None:
            sim_inds = list(range(len(self.sims)))

        # Move the results to a dictionary
        resdict = defaultdict(dict)
        for i,s in enumerate(sim_inds):
            sim = self.sims[s]
            label = sim.label
            if not label: # Give it a label if it doesn't have one
                label = f'Sim {i}'
            if label in resdict: # Avoid duplicates
                label += f' ({i})'
            for reskey in sim.result_keys():
                val = sim.results[reskey].values[t]
                if reskey not in ['r_eff', 'doubling_time']:
                    val = int(val)
                resdict[label][reskey] = val

        # Convert to a dataframe
        df = pd.DataFrame.from_dict(resdict).astype(object) # astype is necessary to prevent type coersion

        if do_plot:
            self.plot_compare(df, *args, **kwargs)

        if output:
            return df
        else:
            print(df)
            return None


    def plot(self, *args, **kwargs):
        ''' Convenience method for plotting -- arguments passed to Sim.plot() '''
        if self.which in ['combined', 'reduced']:
            fig = self.base_sim.plot(*args, **kwargs)
        else:
            fig = None
            orig_setylim = kwargs.get('setylim', True)
            kwargs['legend_args'] = sc.mergedicts({'show_legend':True}, kwargs.get('legend_args')) # Only plot the legend the first time
            for s,sim in enumerate(self.sims):
                if s != 0:
                    kwargs['legend_args']['show_legend'] = False
                if s == len(self.sims)-1:
                    kwargs['setylim'] = orig_setylim
                else:
                    kwargs['setylim'] = False
                fig = sim.plot(fig=fig, *args, **kwargs)
        return fig


    def plot_result(self, key, colors=None, labels=None, *args, **kwargs):
        ''' Convenience method for plotting -- arguments passed to Sim.plot_result() '''
        if self.which in ['combined', 'reduced']:
            fig = self.base_sim.plot_result(key, *args, **kwargs)
        else:
            fig = None
            if colors is None:
                colors = sc.gridcolors(len(self))
            if labels is None:
                labels = [sim.label for sim in self.sims]
            orig_setylim = kwargs.get('setylim', True)
            for s,sim in enumerate(self.sims):
                if s == len(self.sims)-1:
                    kwargs['setylim'] = orig_setylim
                else:
                    kwargs['setylim'] = False
                fig = sim.plot_result(key=key, fig=fig, color=colors[s], label=labels[s], *args, **kwargs)
        return fig


    def plot_compare(self, t=-1, sim_inds=None, log_scale=True, **kwargs):
        '''
        Plot a comparison between sims.

        Args:
            t (int): index of results, passed to compare()
            sim_inds (list): which sims to include, passed to compare()
            log_scale (bool): whether to plot with a logarithmic x-axis
            kwargs (dict): standard plotting arguments, see Sim.plot() for explanation

        Returns:
            fig (figure): the figure handle
        '''
        df = self.compare(t=t, sim_inds=sim_inds, output=True)
        cvplt.plot_compare(df, log_scale=log_scale, **kwargs)

    def save(self, filename=None, keep_people=False, **kwargs):
        '''
        Save to disk as a gzipped pickle. Load with cv.load(filename).

        Args:
            filename (str or None): the name or path of the file to save to; if None, uses default
            keep_people (bool): whether or not to store the population in the Sim objects (NB, very large)
            keywords: passed to makefilepath()

        Returns:
            scenfile (str): the validated absolute path to the saved file

        **Example**::

            msim.save() # Saves to an .msim file
        '''
        if filename is None:
            filename = 'covasim.msim'
        scenfile = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename # Store the actual saved filename

        # Store sims seperately
        sims = self.sims
        self.sims = None # Remove for now

        obj = sc.dcp(self) # This should be quick once we've removed the sims
        if keep_people:
            obj.sims = sims # Just restore the object in full
            print('Note: saving people, which may produce a large file!')
        else:
            obj.base_sim.shrink(in_place=True)
            obj.sims = []
            for sim in sims:
                obj.sims.append(sim.shrink(in_place=False))

        sc.saveobj(filename=scenfile, obj=obj) # Actually save

        self.sims = sims # Restore
        return scenfile


class Scenarios(cvb.ParsObj):
    '''
    Class for running multiple sets of multiple simulations -- e.g., scenarios.

    Args:
        sim (Sim or None): if supplied, use a pre-created simulation as the basis for the scenarios
        metapars (dict): meta-parameters for the run, e.g. number of runs; see make_metapars() for structure
        scenarios (dict): a dictionary defining the scenarios; see default_scenario for structure
        basepars (dict): a dictionary of sim parameters to be used for the basis of the scenarios (not required if sim is provided)
        scenfile (str): a filename for saving (defaults to the creation date)

    Returns:
        scens: a Scenarios object
    '''

    def __init__(self, sim=None, metapars=None, scenarios=None, basepars=None, scenfile=None):

        # For this object, metapars are the foundation
        default_pars = make_metapars() # Start with default pars
        super().__init__(default_pars) # Initialize and set the parameters as attributes

        # Handle filename
        self.created = sc.now()
        if scenfile is None:
            datestr = sc.getdate(obj=self.created, dateformat='%Y-%b-%d_%H.%M.%S')
            scenfile = f'covasim_scenarios_{datestr}.scens'
        self.scenfile = scenfile

        # Handle scenarios -- by default, create a baseline scenario
        if scenarios is None:
            scenarios = {'baseline':{'name':'Baseline', 'pars':{}}}
        self.scenarios = scenarios

        # Handle metapars
        self.metapars = sc.mergedicts({}, metapars)
        self.update_pars(self.metapars)

        # Create the simulation and handle basepars
        if sim is None:
            sim = cvs.Sim()
        self.base_sim = sim
        self.basepars = sc.mergedicts({}, basepars)
        self.base_sim.update_pars(self.basepars)
        self.base_sim.validate_pars()
        self.base_sim.init_results()

        # Copy quantities from the base sim to the main object
        self.npts = self.base_sim.npts
        self.tvec = self.base_sim.tvec

        # Create the results object; order is: results key, scenario, best/low/high
        self.sims = sc.objdict()
        self.results = sc.objdict()
        for reskey in self.result_keys():
            self.results[reskey] = sc.objdict()
            for scenkey in scenarios.keys():
                self.results[reskey][scenkey] = sc.objdict()
                for nblh in ['name', 'best', 'low', 'high']:
                    self.results[reskey][scenkey][nblh] = None # This will get populated below
        return


    def result_keys(self):
        ''' Attempt to retrieve the results keys from the base sim '''
        try:
            keys = self.base_sim.result_keys()
        except Exception as E:
            errormsg = f'Could not retrieve result keys since base sim not accessible: {str(E)}'
            raise ValueError(errormsg)
        return keys


    def run(self, debug=False, keep_people=False, verbose=None, **kwargs):
        '''
        Run the actual scenarios

        Args:
            debug (bool): if True, runs a single run instead of multiple, which makes debugging easier
            verbose (int): level of detail to print, passed to sim.run()
            kwargs (dict): passed to multi_run() and thence to sim.run()

        Returns:
            None (modifies Scenarios object in place)
        '''

        if verbose is None:
            verbose = self['verbose']

        def print_heading(string):
            ''' Choose whether to print a heading, regular text, or nothing '''
            if verbose >= 2:
                sc.heading(string)
            elif verbose == 1:
                print(string)
            return

        reskeys = self.result_keys() # Shorten since used extensively

        # Loop over scenarios
        for scenkey,scen in self.scenarios.items():
            scenname = scen['name']
            scenpars = scen['pars']

            # This is necessary for plotting, and since self.npts is defined prior to run
            if 'n_days' in scenpars.keys():
                errormsg = 'Scenarios cannot be run with different numbers of days; set via basepars instead'
                raise ValueError(errormsg)

            # Create and run the simulations

            print_heading(f'Multirun for {scenkey}')
            scen_sim = sc.dcp(self.base_sim)
            scen_sim.label = scenkey
            scen_sim.update_pars(scenpars)
            run_args = dict(n_runs=self['n_runs'], noise=self['noise'], noisepar=self['noisepar'], keep_people=keep_people, verbose=verbose)
            if debug:
                print('Running in debug mode (not parallelized)')
                run_args.pop('n_runs', None) # Remove n_runs argument, not used for a single run
                scen_sims = [single_run(scen_sim, **run_args, **kwargs)]
            else:
                scen_sims = multi_run(scen_sim, **run_args, **kwargs) # This is where the sims actually get run

            # Process the simulations
            print_heading(f'Processing {scenkey}')

            scenraw = {}
            for reskey in reskeys:
                scenraw[reskey] = np.zeros((self.npts, len(scen_sims)))
                for s,sim in enumerate(scen_sims):
                    scenraw[reskey][:,s] = sim.results[reskey].values

            scenres = sc.objdict()
            scenres.best = {}
            scenres.low = {}
            scenres.high = {}
            for reskey in reskeys:
                scenres.best[reskey] = np.quantile(scenraw[reskey], q=0.5, axis=1) # Changed from median to mean for smoother plots
                scenres.low[reskey]  = np.quantile(scenraw[reskey], q=self['quantiles']['low'], axis=1)
                scenres.high[reskey] = np.quantile(scenraw[reskey], q=self['quantiles']['high'], axis=1)

            for reskey in reskeys:
                self.results[reskey][scenkey]['name'] = scenname
                for blh in ['best', 'low', 'high']:
                    self.results[reskey][scenkey][blh] = scenres[blh][reskey]

            self.sims[scenkey] = scen_sims

        #%% Print statistics
        if verbose:
            sc.heading('Results for last day in each scenario:')
            x = defaultdict(dict)
            scenkeys = list(self.scenarios.keys())
            for scenkey in scenkeys:
                for reskey in reskeys:
                    val = self.results[reskey][scenkey].best[-1]
                    if reskey not in ['r_eff', 'doubling_time']:
                        val = int(val)
                    x[scenkey][reskey] = val
            df = pd.DataFrame.from_dict(x).astype(object)
            print(df)
            print()

        # Save details about the run
        self._kept_people = keep_people

        return


    def plot(self, *args, **kwargs):
        '''
        Plot the results of a scenario.

        Args:
            to_plot      (dict): Dict of results to plot; see get_scen_plots() for structure
            do_save      (bool): Whether or not to save the figure
            fig_path     (str):  Path to save the figure
            fig_args     (dict): Dictionary of kwargs to be passed to pl.figure()
            plot_args    (dict): Dictionary of kwargs to be passed to pl.plot()
            scatter_args (dict): Dictionary of kwargs to be passed to pl.scatter()
            axis_args    (dict): Dictionary of kwargs to be passed to pl.subplots_adjust()
            fill_args    (dict): Dictionary of kwargs to be passed to pl.fill_between()
            legend_args  (dict): Dictionary of kwargs to be passed to pl.legend()
            as_dates     (bool): Whether to plot the x-axis as dates or time points
            dateformat   (str):  Date string format, e.g. '%B %d'
            interval     (int):  Interval between tick marks
            n_cols       (int):  Number of columns of subpanels to use for subplot
            font_size    (int):  Size of the font
            font_family  (str):  Font face
            grid         (bool): Whether or not to plot gridlines
            commaticks   (bool): Plot y-axis with commas rather than scientific notation
            setylim      (bool): Reset the y limit to start at 0
            log_scale    (bool): Whether or not to plot the y-axis with a log scale; if a list, panels to show as log
            do_show      (bool): Whether or not to show the figure
            colors       (dict): Custom color for each scenario, must be a dictionary with one entry per scenario key
            sep_figs     (bool): Whether to show separate figures for different results instead of subplots
            fig          (fig):  Existing figure to plot into

        Returns:
            fig: Figure handle


        **Example**::

            scens = cv.Scenarios()
            scens.run()
            scens.plot()
        '''
        fig = cvplt.plot_scens(scens=self, *args, **kwargs)
        return fig


    def to_json(self, filename=None, tostring=True, indent=2, verbose=False, *args, **kwargs):
        '''
        Export results as JSON.

        Args:
            filename (str): if None, return string; else, write to file

        Returns:
            A unicode string containing a JSON representation of the results,
            or writes the JSON file to disk

        '''
        d = {'t':self.tvec,
             'results':   self.results,
             'basepars':  self.basepars,
             'metapars':  self.metapars,
             'simpars':   self.base_sim.export_pars(),
             'scenarios': self.scenarios
             }
        if filename is None:
            output = sc.jsonify(d, tostring=tostring, indent=indent, verbose=verbose, *args, **kwargs)
        else:
            output = sc.savejson(filename=filename, obj=d, indent=indent, *args, **kwargs)

        return output


    def to_excel(self, filename=None):
        '''
        Export results as XLSX

        Args:
            filename (str): if None, return string; else, write to file

        Returns:
            An sc.Spreadsheet with an Excel file, or writes the file to disk

        '''
        spreadsheet = sc.Spreadsheet()
        spreadsheet.freshbytes()
        with pd.ExcelWriter(spreadsheet.bytes, engine='xlsxwriter') as writer:
            for key in self.result_keys():
                result_df = pd.DataFrame.from_dict(sc.flattendict(self.results[key], sep='_'))
                result_df.to_excel(writer, sheet_name=key)
        spreadsheet.load()

        if filename is None:
            output = spreadsheet
        else:
            output = spreadsheet.save(filename)

        return output


    def save(self, scenfile=None, keep_sims=True, keep_people=False, **kwargs):
        '''
        Save to disk as a gzipped pickle.

        Args:
            scenfile (str or None): the name or path of the file to save to; if None, uses stored
            keep_sims (bool): whether or not to store the actual Sim objects in the Scenarios object
            keep_people (bool): whether or not to store the population in the Sim objects (NB, very large)
            keywords: passed to makefilepath()

        Returns:
            scenfile (str): the validated absolute path to the saved file

        **Example**::

            scens.save() # Saves to a .scens file with the date and time of creation by default

        '''
        if scenfile is None:
            scenfile = self.scenfile
        scenfile = sc.makefilepath(filename=scenfile, **kwargs)
        self.scenfile = scenfile # Store the actual saved filename

        # Store sims seperately
        sims = self.sims
        self.sims = None # Remove for now

        obj = sc.dcp(self) # This should be quick once we've removed the sims
        if not keep_people:
            obj.base_sim.shrink(in_place=True)

        if keep_sims or keep_people:
            if keep_people:
                if not obj._kept_people:
                    print('Warning: there are no people because they were not saved during the run. '
                          'If you want people, please rerun with keep_people=True.')
                obj.sims = sims # Just restore the object in full
                print('Note: saving people, which may produce a large file!')
            else:
                obj.sims = sc.objdict()
                for key in sims.keys():
                    obj.sims[key] = []
                    for sim in sims[key]:
                        obj.sims[key].append(sim.shrink(in_place=False))

        sc.saveobj(filename=scenfile, obj=obj) # Actually save

        self.sims = sims # Restore
        return scenfile


    @staticmethod
    def load(scenfile, *args, **kwargs):
        '''
        Load from disk from a gzipped pickle.

        Args:
            scenfile (str): the name or path of the file to save to
            kwargs: passed to sc.loadobj()

        Returns:
            scens (Scenarios): the loaded scenarios object

        **Example**::

            scens = cv.Scenarios.load('my-scenarios.scens')
        '''
        scens = cvm.load(scenfile, *args, **kwargs)
        if not isinstance(scens, Scenarios):
            errormsg = f'Cannot load object of {type(scens)} as a Scenarios object'
            raise TypeError(errormsg)
        return scens


def single_run(sim, ind=0, reseed=True, noise=0.0, noisepar=None, verbose=None, keep_people=False, run_args=None, sim_args=None, **kwargs):
    '''
    Convenience function to perform a single simulation run. Mostly used for
    parallelization, but can also be used directly.

    Args:
        sim (Sim): the sim instance to be run
        ind (int): the index of this sim
        reseed (bool): whether or not to generate a fresh seed for each run
        noise (float): the amount of noise to add to each run
        noisepar (string): the name of the parameter to add noise to
        verbose (int): detail to print
        run_args (dict): arguments passed to sim.run()
        sim_args (dict): extra parameters to pass to the sim, e.g. 'n_infected'
        kwargs (dict): also passed to the sim

    Returns:
        sim (Sim): a single sim object with results

    **Example**::

        import covasim as cv
        sim = cv.Sim() # Create a default simulation
        sim = cv.single_run(sim) # Run it, equivalent(ish) to sim.run()
    '''

    new_sim = sc.dcp(sim) # Copy the sim to avoid overwriting it

    # Set sim and run arguments
    sim_args = sc.mergedicts(sim_args, kwargs)
    run_args = sc.mergedicts({'verbose':verbose}, run_args)
    if verbose is None:
        verbose = new_sim['verbose']

    if not new_sim.label:
        new_sim.label = f'Sim {ind:d}'

    if reseed:
        new_sim['rand_seed'] += ind # Reset the seed, otherwise no point of parallel runs
        new_sim.set_seed()

    # If the noise parameter is not found, guess what it should be
    if noisepar is None:
        noisepar = 'beta'
        if noisepar not in sim.pars.keys():
            raise sc.KeyNotFoundError(f'Noise parameter {noisepar} was not found in sim parameters')

    # Handle noise -- normally distributed fractional error
    noiseval = noise*np.random.normal()
    if noiseval > 0:
        noisefactor = 1 + noiseval
    else:
        noisefactor = 1/(1-noiseval)
    new_sim[noisepar] *= noisefactor

    if verbose>=1:
        print(f'Running a simulation using {new_sim["rand_seed"]} seed and {noisefactor} noise factor')

    # Handle additional arguments
    for key,val in sim_args.items():
        print(f'Processing {key}:{val}')
        if key in new_sim.pars.keys():
            if verbose>=1:
                print(f'Setting key {key} from {new_sim[key]} to {val}')
                new_sim[key] = val
        else:
            raise sc.KeyNotFoundError(f'Could not set key {key}: not a valid parameter name')

    # Run
    new_sim.run(**run_args)

    # Shrink the sim to save memory
    if not keep_people:
        new_sim.shrink()

    return new_sim


def multi_run(sim, n_runs=4, reseed=True, noise=0.0, noisepar=None, iterpars=None, verbose=None, combine=False, keep_people=None, run_args=None, sim_args=None, par_args=None, **kwargs):
    '''
    For running multiple runs in parallel. If the first argument is a list of sims,
    exactly these will be run and most other arguments will be ignored.

    Args:
        sim (Sim or list): the sim instance to be run, or a list of sims.
        n_runs (int): the number of parallel runs
        reseed (bool): whether or not to generate a fresh seed for each run
        noise (float): the amount of noise to add to each run
        noisepar (string): the name of the parameter to add noise to
        iterpars (dict): any other parameters to iterate over the runs; see sc.parallelize() for syntax
        verbose (int): detail to print
        combine (bool): whether or not to combine all results into one sim, rather than return multiple sim objects
        keep_people (bool): whether or not to keep the people in each sim
        run_args (dict): arguments passed to sim.run()
        sim_args (dict): extra parameters to pass to the sim
        par_args (dict): arguments passed to sc.parallelize()
        kwargs (dict): also passed to the sim

    Returns:
        If combine is True, a single sim object with the combined results from each sim.
        Otherwise, a list of sim objects (default).

    **Example**::

        import covasim as cv
        sim = cv.Sim()
        sims = cv.multi_run(sim, n_runs=6, noise=0.2)
    '''

    # Handle inputs
    sim_args = sc.mergedicts(sim_args, kwargs) # Handle blank
    par_args = sc.mergedicts(par_args) # Handle blank

    # Handle iterpars
    if iterpars is None:
        iterpars = {}
    else:
        n_runs = None # Reset and get from length of dict instead
        for key,val in iterpars.items():
            new_n = len(val)
            if n_runs is not None and new_n != n_runs:
                raise ValueError(f'Each entry in iterpars must have the same length, not {n_runs} and {len(val)}')
            else:
                n_runs = new_n

    # Run the sims
    if isinstance(sim, cvs.Sim): # Normal case: one sim
        iterkwargs = {'ind':np.arange(n_runs)}
        iterkwargs.update(iterpars)
        kwargs = dict(sim=sim, reseed=reseed, noise=noise, noisepar=noisepar, verbose=verbose, keep_people=keep_people, sim_args=sim_args, run_args=run_args)
        sims = sc.parallelize(single_run, iterkwargs=iterkwargs, kwargs=kwargs, **par_args)
    elif isinstance(sim, list): # List of sims
        iterkwargs = {'sim':sim}
        kwargs = dict(verbose=verbose, keep_people=keep_people, sim_args=sim_args, run_args=run_args)
        sims = sc.parallelize(single_run, iterkwargs=iterkwargs, kwargs=kwargs, **par_args)
    else:
        errormsg = f'Must be Sim object or list, not {type(sim)}'
        raise TypeError(errormsg)

    return sims
