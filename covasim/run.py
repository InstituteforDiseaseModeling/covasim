'''
Functions for running multiple Covasim runs.
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import pylab as pl
import sciris as sc


# Specify all externally visible functions this file defines
__all__ = ['make_metapars', 'Scenarios', 'single_run', 'multi_run']

def make_metapars():
    ''' Create default metaparameters for a Scenarios run '''
    metapars = dict(
        n = 3, # Number of parallel runs; change to 3 for quick, 11 for real
        noise = 0.1, # Use noise, optionally
        noisepar = 'beta',
        seed = 1,
        reskeys = ['cum_exposed', 'n_exposed'],
        quantiles = {'low':0.1, 'high':0.9},
    )
    return metapars


class Scenarios(sc.prettyobj):
    '''
    Class for running multiple sets of multiple simulations -- e.g., scenarios
    '''

    def __init__(self, filename=None):
        self.created = sc.now()
        if filename is None:
            datestr = sc.getdate(obj=self.created, dateformat='%Y-%b-%d_%H.%M.%S')
            filename = f'covasim_{datestr}.sim'
        self.filename = filename

        # Order is: results key, scenario, best/low/high
        self.allres = sc.objdict()
        for reskey in reskeys:
            self.allres[reskey] = sc.objdict()
            for scenkey in scenarios.keys():
                self.allres[reskey][scenkey] = sc.objdict()
                for nblh in ['name', 'best', 'low', 'high']:
                    self.allres[reskey][scenkey][nblh] = None # This will get populated below
        return

    def run(self):



        for scenkey,scenname in scenarios.items():

            scen_sim = cova.Sim()
            scen_sim.set_seed(seed)

            if scenkey == 'baseline':
                scen_sim['interv_days'] = [] # No interventions
                scen_sim['interv_effs'] = []

            elif scenkey == 'distance':
                scen_sim['interv_days'] = [interv_day] # Close schools for 2 weeks starting Mar. 16, then reopen
                scen_sim['interv_effs'] = [0.7] # Change to 40% and then back to 70%

            elif scenkey == 'isolatepos':
                scen_sim['diag_factor'] = 0.1 # Scale beta by this amount for anyone who's diagnosed

            else:
                raise KeyError


            sc.heading(f'Multirun for {scenkey}')

            scen_sims = cova.multi_run(scen_sim, n=n, noise=noise, noisepar=noisepar, verbose=verbose)

            sc.heading(f'Processing {scenkey}')

            # TODO: this only needs to be done once
            res0 = scen_sims[0].results
            npts = res0[reskeys[0]].npts
            tvec = res0['t']

            scenraw = {}
            for reskey in reskeys:
                scenraw[reskey] = pl.zeros((npts, n))
                for s,sim in enumerate(scen_sims):
                    scenraw[reskey][:,s] = sim.results[reskey].values

            scenres = sc.objdict()
            scenres.best = {}
            scenres.low = {}
            scenres.high = {}
            for reskey in reskeys:
                scenres.best[reskey] = pl.mean(scenraw[reskey], axis=1) # Changed from median to mean for smoother plots
                scenres.low[reskey]  = pl.quantile(scenraw[reskey], q=quantiles['low'], axis=1)
                scenres.high[reskey] = pl.quantile(scenraw[reskey], q=quantiles['high'], axis=1)

            for reskey in reskeys:
                allres[reskey][scenkey]['name'] = scenname
                for blh in ['best', 'low', 'high']:
                    allres[reskey][scenkey][blh] = scenres[blh][reskey]

            if save_sims:
                print('WARNING: saving sims, which will produce a very large file!')
                allres['sims'] = scen_sims

        #%% Print statistics
        for reskey in reskeys:
            for scenkey in list(scenarios.keys()):
                print(f'{reskey} {scenkey}: {allres[reskey][scenkey].best[-1]:0.0f}')

        # Perform health systems analysis
        hsys = covidhs.HealthSystem(allres)
        hsys.analyze()
        hsys.plot()

        return


    def plot(self, do_save=None, fig_path=None, fig_args=None, plot_args=None,
             axis_args=None, as_dates=True, interval=None, dateformat=None,
             font_size=18, font_family=None, use_grid=True, do_show=True, verbose=None):
        '''
        Plot the results -- can supply arguments for both the figure and the plots.

        Args:
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
        for rk, reskey in enumerate(reskeys):
            pl.subplot(len(reskeys), 1, rk + 1)

            resdata = allres[reskey]

            for scenkey, scendata in resdata.items():
                pl.fill_between(tvec, scendata.low, scendata.high, **fill_args)
                pl.plot(tvec, scendata.best, label=scendata.name, **plot_args)

                # interv_col = [0.5, 0.2, 0.4]

                ymax = pl.ylim()[1]

                if reskey == 'cum_exposed':
                    sc.setylim()
                    pl.title('Cumulative infections')
                    pl.text(0.0, 1.1, 'COVID-19 projections, per 1 million susceptibles', fontsize=24,
                            transform=pl.gca().transAxes)

                elif reskey == 'n_exposed':
                    pl.legend()
                    sc.setylim()
                    pl.title('Active infections')

                pl.grid(True)

                # Set x-axis
                xt = pl.gca().get_xticks()
                lab = []
                for t in xt:
                    tmp = dt.datetime(2020, 1, 1) + dt.timedelta(days=int(t))  # + pars['day_0']
                    lab.append(tmp.strftime('%b-%d'))
                pl.gca().set_xticklabels(lab)
                sc.commaticks(axis='y')

        if do_save:
            pl.savefig(fig_path, dpi=150)
            if do_run:  # Don't resave loaded data
                sc.saveobj(obj_path, allres)

        if show_plot: # Optionally show plot
            pl.show()

        return fig


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