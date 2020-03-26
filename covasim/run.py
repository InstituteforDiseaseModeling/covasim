'''
Functions for running multiple Covasim runs.
'''

#%% Imports
import numpy as np
import pylab as pl
import sciris as sc
from . import base as cvbase
from . import model as cvmodel
import covid_healthsystems as covidhs


# Specify all externally visible functions this file defines
__all__ = ['default_scen_plots', 'make_metapars', 'Scenarios', 'single_run', 'multi_run']


default_scen_plots = sc.odict({
            'cum_exposed': 'Cumulative infections',
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


def make_metapars():
    ''' Create default metaparameters for a Scenarios run '''
    metapars = dict(
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
    Class for running multiple sets of multiple simulations -- e.g., scenarios
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
            scenarios = {'baseline':{'name':'Baseline', 'pars':{}}}
        self.scenarios = scenarios

        # Handle metapars
        if metapars is None:
            metapars = {}
        self.metapars = metapars
        self.update_pars(self.metapars)

        # Create the simulation and handle basepars
        if sim is None:
            sim = cvmodel.Sim()
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
        self.allres = sc.objdict()
        for reskey in self.reskeys:
            self.allres[reskey] = sc.objdict()
            for scenkey in scenarios.keys():
                self.allres[reskey][scenkey] = sc.objdict()
                for nblh in ['name', 'best', 'low', 'high']:
                    self.allres[reskey][scenkey][nblh] = None # This will get populated below
        return


    def run(self, keep_sims=False, verbose=None):
        ''' Run the actual scenarios'''

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
            scen_sim = cvmodel.Sim(pars=self.basepars)
            scen_sim.update_pars(scenpars)
            scen_sims = multi_run(scen_sim, n=self['n_runs'], noise=self['noise'], noisepar=self['noisepar'], verbose=verbose)

            # Process the simulations
            print_heading(f'Processing {scenkey}')

            scenraw = {}
            for reskey in reskeys:
                scenraw[reskey] = pl.zeros((self.npts, self['n_runs']))
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

            if keep_sims:
                print('WARNING: saving sims, which will produce a very large file!')
                self.allres['sims'] = scen_sims
                sc.checkmem(self.allres) # Print a warning about how big the file is likely to be


        #%% Print statistics
        if verbose:
            for reskey in reskeys:
                for scenkey in list(self.scenarios.keys()):
                    print(f'{reskey} {scenkey}: {self.allres[reskey][scenkey].best[-1]:0.0f}')

        # Perform health systems analysis
        self.hsys = covidhs.HealthSystem(self.allres)
        self.hsys.analyze()

        return


    def plot(self, to_plot=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
             axis_args=None, as_dates=True, interval=None, dateformat=None,
             font_size=18, font_family=None, use_grid=True, do_show=True, verbose=None):
        '''
        Plot the results -- can supply arguments for both the figure and the plots.

        Args:
            to_plot (dict): Dict of results to plot; see default_scen_plots for structure
            do_save (bool or str): Whether or not to save the figure. If a string, save to that filename.
            fig_path (str): Path to save the figure
            fig_args (dict): Dictionary of kwargs to be passed to pl.figure()
            plot_args (dict): Dictionary of kwargs to be passed to pl.plot()
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
        sc.printv('Plotting...', 1, verbose)

        if to_plot is None:
            to_plot = default_scen_plots
        to_plot = sc.odict(to_plot) # In case it's supplied as a dict

        fig_args = {'figsize': (16, 12)}
        plot_args = {'lw': 3, 'alpha': 0.7}
        axis_args = {'left': 0.10, 'bottom': 0.05, 'right': 0.95, 'top': 0.90, 'wspace': 0.5, 'hspace': 0.25}
        fill_args = {'alpha': 0.2}
        font_size = 18

        fig = pl.figure(**fig_args)
        pl.subplots_adjust(**axis_args)
        pl.rcParams['font.size'] = font_size
        pl.rcParams['font.family'] = 'Proxima Nova' # NB, may not be available on all systems

        # %% Plotting
        for rk,reskey,title in to_plot.enumitems():
            ax = pl.subplot(len(to_plot), 1, rk + 1)

            resdata = self.allres[reskey]

            for scenkey, scendata in resdata.items():
                pl.fill_between(self.tvec, scendata.low, scendata.high, **fill_args)
                pl.plot(self.tvec, scendata.best, label=scendata.name, **plot_args)

                pl.title(title)
                if rk == 0:
                    pl.legend(loc='best')

                sc.setylim()
                pl.grid(use_grid)

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


    def plot_healthsystem(self, *args, **kwargs):
        ''' Very simple method to plot the health system results '''
        return self.hsys.plot(*args, **kwargs)


    def save(self, filename=None, **kwargs):
        '''
        Save to disk as a gzipped pickle.

        Args:
            filename (str or None): the name or path of the file to save to; if None, uses stored
            keywords: passed to makefilepath()

        Returns:
            filename (str): the validated absolute path to the saved file

        Example:
            scens.save() # Saves to a .scens file with the date and time of creation by default

        '''
        if filename is None:
            filename = self.filename
        filename = sc.makefilepath(filename=filename, **kwargs)
        sc.saveobj(filename=filename, obj=self)
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



def single_run(sim, ind=0, noise=0.0, noisepar=None, verbose=None, sim_args=None, **kwargs):
    '''
    Convenience function to perform a single simulation run. Mostly used for
    parallelization, but can also be used directly:
        import covasim.cova_generic as cova
        sim = cova.Sim() # Create a default simulation
        sim = cova.single_run(sim) # Run it, equivalent(ish) to sim.run()
    '''

    if sim_args is None:
        sim_args = {}

    new_sim = sc.dcp(sim) # To avoid overwriting it; otherwise, use

    if verbose is None:
        verbose = new_sim['verbose']

    new_sim['seed'] += ind # Reset the seed, otherwise no point of parallel runs
    new_sim.set_seed()

    # If the noise parameter is not found, guess what it should be
    if noisepar is None:
        guesses = ['r_contact', 'r0', 'beta']
        found = [guess for guess in guesses if guess in sim.pars.keys()]
        if len(found)!=1:
            raise KeyError(f'Cound not guess noise parameter since out of {guesses}, {found} were found')
        else:
            noisepar = found[0]

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
    for key,val in kwargs.items():
        print(f'Processing {key}:{val}')
        if key in new_sim.pars.keys():
            if verbose>=1:
                print(f'Setting key {key} from {new_sim[key]} to {val}')
                new_sim[key] = val
            pass
        else:
            raise KeyError(f'Could not set key {key}: not a valid parameter name')

    # Run
    new_sim.run(verbose=verbose)

    return new_sim


def multi_run(sim, n=4, noise=0.0, noisepar=None, iterpars=None, verbose=None, sim_args=None, combine=False, **kwargs):
    '''
    For running multiple runs in parallel. Example:
        import covid_seattle
        sim = covid_seattle.Sim()
        sims = covid_seattle.multi_run(sim, n=6, noise=0.2)
    '''

    # Create the sims
    if sim_args is None:
        sim_args = {}

    # Handle iterpars
    if iterpars is None:
        iterpars = {}
    else:
        n = None # Reset and get from length of dict instead
        for key,val in iterpars.items():
            new_n = len(val)
            if n is not None and new_n != n:
                raise ValueError(f'Each entry in iterpars must have the same length, not {n} and {len(val)}')
            else:
                n = new_n

    # Copy the simulations
    iterkwargs = {'ind':np.arange(n)}
    iterkwargs.update(iterpars)
    kwargs = {'sim':sim, 'noise':noise, 'noisepar':noisepar, 'verbose':verbose, 'sim_args':sim_args}
    sims = sc.parallelize(single_run, iterkwargs=iterkwargs, kwargs=kwargs)

    if not combine:
        output = sims
    else:
        print('WARNING: not tested!')
        output_sim = sc.dcp(sims[0])
        output_sim.pars['parallelized'] = n # Store how this was parallelized
        output_sim.pars['n'] *= n # Restore this since used in later calculations -- a bit hacky, it's true
        for sim in sims[1:]: # Skip the first one
            output_sim.people.update(sim.people)
            for key in sim.results_keys:
                output_sim.results[key] += sim.results[key]
        output = output_sim

    return output