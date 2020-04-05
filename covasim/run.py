'''
Functions for running multiple Covasim runs.
'''

#%% Imports
import numpy as np
import pylab as pl
import sciris as sc
from . import base as cvbase
from . import sim as cvsim


# Specify all externally visible functions this file defines
__all__ = ['default_scen_plots', 'default_scenario', 'make_metapars', 'Scenarios', 'single_run', 'multi_run']


default_scen_plots = sc.odict({
            'cum_infections': 'Cumulative infections',
            # 'cum_deaths': 'Cumulative deaths',
            # 'cum_recoveries':'Cumulative recoveries',
            # 'cum_tested': 'Cumulative tested',
            # 'n_susceptible': 'Number susceptible',
            'n_infectious': 'Number of active infections',
            # 'cum_diagnosed': 'Cumulative diagnosed',
            # 'infections': 'New infections',
            # 'deaths': 'New deaths',
            # 'recoveries': 'New recoveries',
            # 'tests': 'Number of tests',
            # 'diagnoses': 'New diagnoses',
    })

default_scenario = {'baseline':{'name':'Baseline', 'pars':{}}}


def make_metapars():
    ''' Create default metaparameters for a Scenarios run '''
    metapars = sc.objdict(
        n_runs    = 3, # Number of parallel runs; change to 3 for quick, 11 for real
        noise     = 0.1, # Use noise, optionally
        noisepar  = 'beta',
        seed      = 1,
        quantiles = {'low':0.1, 'high':0.9},
        verbose   = 1,
    )
    return metapars


class Scenarios(cvbase.ParsObj):
    '''
    Class for running multiple sets of multiple simulations -- e.g., scenarios.

    Args:
        sim (Sim or None): if supplied, use a pre-created simulation as the basis for the scenarios
        metapars (dict): meta-parameters for the run, e.g. number of runs; see make_metapars() for structure
        scenarios (dict): a dictionary defining the scenarios; see default_scenario for structure
        basepars (dict): a dictionary of sim parameters to be used for the basis of the scenarios (not required if sim is provided)
        filename (str): a filename for saving (defaults to the creation date)

    Returns:
        scens: a Scenarios object
    '''

    def __init__(self, sim=None, metapars=None, scenarios=None, basepars=None, filename=None):

        # For this object, metapars are the foundation
        default_pars = make_metapars() # Start with default pars
        super().__init__(default_pars) # Initialize and set the parameters as attributes

        # Handle filename
        self.created = sc.now()
        if filename is None:
            datestr = sc.getdate(obj=self.created, dateformat='%Y-%b-%d_%H.%M.%S')
            filename = f'covasim_scenarios_{datestr}.scens'
        self.filename = filename

        # Handle scenarios -- by default, create a baseline scenario
        if scenarios is None:
            scenarios = sc.dcp(default_scenario)
        self.scenarios = scenarios

        # Handle metapars
        if metapars is None:
            metapars = {}
        self.metapars = metapars
        self.update_pars(self.metapars)

        # Create the simulation and handle basepars
        if sim is None:
            sim = cvsim.Sim()
        self.base_sim = sim
        if basepars is None:
            basepars = {}
        self.basepars = basepars
        self.base_sim.update_pars(self.basepars)
        self.base_sim.validate_pars()
        self.base_sim.init_results()

        # Copy quantities from the base sim to the main object
        self.npts = self.base_sim.npts
        self.tvec = self.base_sim.tvec
        self.reskeys = self.base_sim.reskeys

        # Create the results object; order is: results key, scenario, best/low/high
        self.sims = sc.objdict()
        self.allres = sc.objdict()
        for reskey in self.reskeys:
            self.allres[reskey] = sc.objdict()
            for scenkey in scenarios.keys():
                self.allres[reskey][scenkey] = sc.objdict()
                for nblh in ['name', 'best', 'low', 'high']:
                    self.allres[reskey][scenkey][nblh] = None # This will get populated below
        return


    def run(self, debug=False, verbose=None, **kwargs):
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

        reskeys = self.reskeys # Shorten since used extensively

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
            scen_sim.update_pars(scenpars)
            run_args = dict(n_runs=self['n_runs'], noise=self['noise'], noisepar=self['noisepar'], verbose=verbose)
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
                scenraw[reskey] = pl.zeros((self.npts, len(scen_sims)))
                for s,sim in enumerate(scen_sims):
                    scenraw[reskey][:,s] = sim.results[reskey].values

            scenres = sc.objdict()
            scenres.best = {}
            scenres.low = {}
            scenres.high = {}
            for reskey in reskeys:
                scenres.best[reskey] = pl.mean(scenraw[reskey], axis=1) # Changed from median to mean for smoother plots
                scenres.low[reskey]  = pl.quantile(scenraw[reskey], q=self['quantiles']['low'], axis=1)
                scenres.high[reskey] = pl.quantile(scenraw[reskey], q=self['quantiles']['high'], axis=1)

            for reskey in reskeys:
                self.allres[reskey][scenkey]['name'] = scenname
                for blh in ['best', 'low', 'high']:
                    self.allres[reskey][scenkey][blh] = scenres[blh][reskey]

            self.sims[scenkey] = scen_sims



        #%% Print statistics
        if verbose:
            print('\nResults for final time point in each scenario:')
            for reskey in reskeys:
                print(f'\n{reskey}')
                for scenkey in list(self.scenarios.keys()):
                    print(f'  {scenkey}: {self.allres[reskey][scenkey].best[-1]:0.0f}')
            print() # Add a blank space

        return


    def plot(self, to_plot=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
             axis_args=None, fill_args=None, as_dates=True, interval=None, dateformat=None,
             font_size=18, font_family=None, grid=True, commaticks=True, do_show=True, sep_figs=False,
             verbose=None):
        '''
        Plot the results -- can supply arguments for both the figure and the plots.

        Args:
            to_plot     (dict): Dict of results to plot; see default_scen_plots for structure
            do_save     (bool): Whether or not to save the figure
            fig_path    (str):  Path to save the figure
            fig_args    (dict): Dictionary of kwargs to be passed to pl.figure()
            plot_args   (dict): Dictionary of kwargs to be passed to pl.plot()
            axis_args   (dict): Dictionary of kwargs to be passed to pl.subplots_adjust()
            fill_args   (dict): Dictionary of kwargs to be passed to pl.fill_between()
            as_dates    (bool): Whether to plot the x-axis as dates or time points
            interval    (int):  Interval between tick marks
            dateformat  (str):  Date string format, e.g. '%B %d'
            font_size   (int):  Size of the font
            font_family (str):  Font face
            grid        (bool): Whether or not to plot gridlines
            commaticks  (bool): Plot y-axis with commas rather than scientific notation
            do_show     (bool): Whether or not to show the figure
            sep_figs    (bool): Whether to show separate figures for different results instead of subplots
            verbose     (bool): Display a bit of extra information

        Returns:
            fig: Figure handle
        '''

        if verbose is None:
            verbose = self['verbose']
        sc.printv('Plotting...', 1, verbose)

        if to_plot is None:
            to_plot = default_scen_plots
        to_plot = sc.odict(sc.dcp(to_plot)) # In case it's supplied as a dict

        # Handle input arguments -- merge user input with defaults
        fig_args  = sc.mergedicts({'figsize': (16, 12)}, fig_args)
        plot_args = sc.mergedicts({'lw': 3, 'alpha': 0.7}, plot_args)
        axis_args = sc.mergedicts({'left': 0.10, 'bottom': 0.05, 'right': 0.95, 'top': 0.90, 'wspace': 0.5, 'hspace': 0.25}, axis_args)
        fill_args = sc.mergedicts({'alpha': 0.2}, fill_args)

        if sep_figs:
            figs = []
        else:
            fig = pl.figure(**fig_args)
        pl.subplots_adjust(**axis_args)
        pl.rcParams['font.size'] = font_size
        if font_family:
            pl.rcParams['font.family'] = font_family

        # %% Plotting
        for rk,reskey,title in to_plot.enumitems():
            if sep_figs:
                figs.append(pl.figure(**fig_args))
                ax = pl.subplot(111)
            else:
                ax = pl.subplot(len(to_plot), 1, rk + 1)

            resdata = self.allres[reskey]

            for scenkey, scendata in resdata.items():

                pl.fill_between(self.tvec, scendata.low, scendata.high, **fill_args)
                pl.plot(self.tvec, scendata.best, label=scendata.name, **plot_args)
                pl.title(title)
                if rk == 0:
                    pl.legend(loc='best')

                pl.grid(grid)
                if commaticks:
                    sc.commaticks()

                # Optionally reset tick marks (useful for e.g. plotting weeks/months)
                if interval:
                    xmin,xmax = ax.get_xlim()
                    ax.set_xticks(pl.arange(xmin, xmax+1, interval))

                # Set xticks as dates
                if as_dates:
                    xticks = ax.get_xticks()
                    xticklabels = self.base_sim.inds2dates(xticks, dateformat=dateformat)
                    ax.set_xticklabels(xticklabels)

        # Ensure the figure actually renders or saves
        if do_save:
            if fig_path is None: # No figpath provided - see whether do_save is a figpath
                fig_path = 'covasim_scenarios.png' # Just give it a default name
            fig_path = sc.makefilepath(fig_path) # Ensure it's valid, including creating the folder
            pl.savefig(fig_path)

        if do_show:
            pl.show()
        else:
            pl.close(fig)

        return fig



    def save(self, filename=None, keep_sims=True, keep_people=False, **kwargs):
        '''
        Save to disk as a gzipped pickle.

        Args:
            filename (str or None): the name or path of the file to save to; if None, uses stored
            keep_sims (bool): whether or not to store the actual Sim objects in the Scenarios object
            keep_people (bool): whether or not to store the people in the Sim objects (NB, very large)
            keywords: passed to makefilepath()

        Returns:
            filename (str): the validated absolute path to the saved file

        Example:
            scens.save() # Saves to a .scens file with the date and time of creation by default

        '''
        if filename is None:
            filename = self.filename
        filename = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename # Store the actual saved filename

        # Store sims seperately
        sims = self.sims
        self.sims = None # Remove for now

        obj = sc.dcp(self) # This should be quick once we've removed the sims

        if keep_sims:
            if keep_people:
                obj.sims = sims # Just restore the object in full
                print('Note: saving people, which may produce a large file!')
            else:
                obj.sims = sc.objdict()
                for key in sims.keys():
                    obj.sims[key] = []
                    for sim in sims[key]:
                        obj.sims[key].append(sim.shrink())

        sc.saveobj(filename=filename, obj=obj) # Actually save

        self.sims = sims # Restore
        return filename


    @staticmethod
    def load(filename, **kwargs):
        '''
        Load from disk from a gzipped pickle.

        Args:
            filename (str): the name or path of the file to save to
            keywords: passed to makefilepath()

        Returns:
            scens (Scenarios): the loaded scenarios object

        Example:
            sim = cv.Scenarios.load('my-scenarios.scens')
        '''
        filename = sc.makefilepath(filename=filename, **kwargs)
        scens = sc.loadobj(filename=filename)
        return scens



def single_run(sim, ind=0, noise=0.0, noisepar=None, verbose=None, run_args=None, sim_args=None, **kwargs):
    '''
    Convenience function to perform a single simulation run. Mostly used for
    parallelization, but can also be used directly.

    Args:
        sim (Sim): the sim instance to be run
        ind (int): the index of this sim
        noise (float): the amount of noise to add to each run
        noisepar (string): the name of the parameter to add noise to
        verbose (int): detail to print
        run_args (dict): arguments passed to sim.run()
        sim_args (dict): extra parameters to pass to the sim, e.g. 'n_infected'
        kwargs (dict): also passed to the sim

    Returns:
        sim (Sim): a single sim object with results

    Example:
        import covasim as cv
        sim = cv.Sim() # Create a default simulation
        sim = cv.single_run(sim) # Run it, equivalent(ish) to sim.run()
    '''

    new_sim = sc.dcp(sim) # Copy the sim to avoid overwriting it

    # Set sim and run arguments
    if verbose is None:
        verbose = new_sim['verbose']
    sim_args = sc.mergedicts(sim_args, kwargs)
    run_args = sc.mergedicts({'verbose':verbose}, run_args)

    new_sim['seed'] += ind # Reset the seed, otherwise no point of parallel runs
    new_sim.set_seed()

    # If the noise parameter is not found, guess what it should be
    if noisepar is None:
        noisepar = 'beta'
        if noisepar not in sim.pars.keys():
            raise KeyError(f'Noise parameter {noisepar} was not found in sim parameters')

    # Handle noise -- normally distributed fractional error
    noiseval = noise*np.random.normal()
    if noiseval > 0:
        noisefactor = 1 + noiseval
    else:
        noisefactor = 1/(1-noiseval)
    new_sim[noisepar] *= noisefactor

    if verbose>=1:
        print(f'Running a simulation using {new_sim["seed"]} seed and {noisefactor} noise')

    # Handle additional arguments
    for key,val in sim_args.items():
        print(f'Processing {key}:{val}')
        if key in new_sim.pars.keys():
            if verbose>=1:
                print(f'Setting key {key} from {new_sim[key]} to {val}')
                new_sim[key] = val
        else:
            raise KeyError(f'Could not set key {key}: not a valid parameter name')

    # Run
    new_sim.run(**run_args)

    return new_sim


def multi_run(sim, n_runs=4, noise=0.0, noisepar=None, iterpars=None, verbose=None, combine=False, run_args=None, sim_args=None, **kwargs):
    '''
    For running multiple runs in parallel.

    Args:
        sim (Sim): the sim instance to be run
        n_runs (int): the number of parallel runs
        noise (float): the amount of noise to add to each run
        noisepar (string): the name of the parameter to add noise to
        iterpars (dict): any other parameters to iterate over the runs; see sc.parallelize() for syntax
        verbose (int): detail to print
        combine (bool): whether or not to combine all results into one sim, rather than return multiple sim objects
        run_args (dict): arguments passed to sim.run()
        sim_args (dict): extra parameters to pass to the sim
        kwargs (dict): also passed to the sim

    Returns:
        if combine:
            a single sim object with the combined results from each sim
        else (default):
            a list of sim objects

    Example:
        import covasim as cv
        sim = cv.Sim()
        sims = cv.multi_run(sim, n_runs=6, noise=0.2)
    '''

    # Create the sims
    if sim_args is None:
        sim_args = {}

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

    # Copy the simulations
    iterkwargs = {'ind':np.arange(n_runs)}
    iterkwargs.update(iterpars)
    kwargs = {'sim':sim, 'noise':noise, 'noisepar':noisepar, 'verbose':verbose, 'sim_args':sim_args, 'run_args':run_args}
    sims = sc.parallelize(single_run, iterkwargs=iterkwargs, kwargs=kwargs)

    # Usual case -- return a list of sims
    if not combine:
        return sims

    # Or, combine them into a single sim with scaled results
    else:
        output_sim = sc.dcp(sims[0])
        output_sim.pars['parallelized'] = n_runs # Store how this was parallelized
        output_sim.pars['n'] *= n_runs # Restore this since used in later calculations -- a bit hacky, it's true
        for s,sim in enumerate(sims[1:]): # Skip the first one
            output_sim.people.update(sim.people)
            for key in sim.reskeys:
                this_res = sim.results[key]
                output_sim.results[key].values += this_res.values

        # For non-count results (scale=False), rescale them
        for key in output_sim.reskeys:
            if not output_sim.results[key].scale:
                output_sim.results[key].values /= len(sims)

        return output_sim