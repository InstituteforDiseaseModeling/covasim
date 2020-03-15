'''
This file contains all the code for a single run of Covid-ABM.

Based heavily on LEMOD-FP (https://github.com/amath-idm/lemod_fp).
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import sciris as sc
from . import utils as cov_ut

# Specify all externally visible functions this file defines
__all__ = ['ParsObj', 'Person', 'Sim', 'single_run', 'multi_run']



#%% Define classes
class ParsObj(sc.prettyobj):
    '''
    A class based around performing operations on a self.pars dict.
    '''

    def __init__(self, pars):
        self.update_pars(pars)
        return

    def __getitem__(self, key):
        ''' Allow sim['par_name'] instead of sim.pars['par_name'] '''
        return self.pars[key]

    def __setitem__(self, key, value):
        ''' Ditto '''
        if key in self.pars:
            self.pars[key] = value
        else:
            suggestion = sc.suggest(key, self.pars.keys())
            if suggestion:
                errormsg = f'Key {key} not found; did you mean "{suggestion}"?'
            else:
                all_keys = '\n'.join(list(self.pars.keys()))
                errormsg = f'Key {key} not found; available keys:\n{all_keys}'
            raise KeyError(errormsg)
        return

    def update_pars(self, pars):
        ''' Update internal dict with new pars '''
        if not isinstance(pars, dict):
            raise TypeError(f'The pars object must be a dict; you supplied a {type(pars)}')
        if not hasattr(self, 'pars'):
            self.pars = pars
        elif pars is not None:
            self.pars.update(pars)
        return


class Person(ParsObj):
    '''
    Class for a single person.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Initialize and set the parameters as attributes
        return


class Sim(ParsObj):
    '''
    The Sim class handles the running of the simulation: the number of children,
    number of time points, and the parameters of the simulation.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Initialize and set the parameters as attributes
        return

    def set_seed(self, seed=None, reset=False):
        ''' Set the seed for the random number stream '''
        if reset:
            seed = self['seed']
        cov_ut.set_seed(seed)
        return

    @property
    def n(self):
        ''' Count the number of people '''
        return len(self.people)

    @property
    def npts(self):
        ''' Count the number of time points '''
        return int(self['n_days'] + 1)

    @property
    def tvec(self):
        ''' Create a time vector '''
        return np.arange(self['n_days'] + 1)


    def get_person(self, ind):
        ''' Return a person based on their ID '''
        return self.people[self.uids[ind]]


    def init_results(self):
        ''' Initialize results '''
        raise NotImplementedError


    def init_people(self):
        ''' Create the people '''
        raise NotImplementedError


    def summary_stats(self):
        ''' Compute the summary statistics to display at the end of a run '''
        raise NotImplementedError


    def run(self):
        ''' Run the simulation '''
        raise NotImplementedError


    def likelihood(self):
        '''
        Compute the log-likelihood of the current simulation based on the number
        of new diagnoses.
        '''
        raise NotImplementedError



    def plot(self):
        '''
        Plot the results -- can supply arguments for both the figure and the plots.

        Parameters
        ----------
        do_save : bool or str
            Whether or not to save the figure. If a string, save to that filename.

        fig_args : dict
            Dictionary of kwargs to be passed to pl.figure()

        plot_args : dict
            Dictionary of kwargs to be passed to pl.plot()

        as_days : bool
            Whether to plot the x-axis as days or time points

        Returns
        -------
        Figure handle
        '''
        raise NotImplementedError


    def plot_people(self):
        ''' Use imshow() to show all individuals as rows, with time as columns, one pixel per timestep per person '''
        raise NotImplementedError


def single_run(sim=None, ind=0, noise=0.0, noisepar=None, verbose=None, **kwargs):
    '''
    Convenience function to perform a single simulation run. Mostly used for
    parallelization, but can also be used directly:
        import covid_abm
        sim = covid_abm.single_run() # Create and run a default simulation
    '''
    if sim is None:
        new_sim = Sim(**kwargs)
    else:
        new_sim = sc.dcp(sim) # To avoid overwriting it; otherwise, use

    noisefactor = np.maximum(0, 1+noise*np.random.normal()) # Optionally add noise
    new_sim['seed'] += ind # Reset the seed, otherwise no point of parallel runs
    new_sim.set_seed(new_sim['seed'])
    new_sim[noisepar] *= noisefactor
    new_sim.run(verbose=verbose)

    return new_sim


def multi_run(sim=None, n=4, noise=0.0, noisepar=None, verbose=None, combine=False, **kwargs):
    '''
    For running multiple runs in parallel. Example:
        import covid_seattle
        sim = covid_seattle.Sim()
        sims = covid_seattle.multi_run(sim, n=6, noise=0.2)
    '''
    if sim is None:
        sim = Sim(**kwargs)

    # If the noise parameter is not found, guess what it should be
    if noisepar is None:
        guesses = ['r_contact', 'r0', 'beta']
        found = [guess for guess in guesses if guess in sim.pars.keys()]
        if len(found)!=1:
            raise KeyError(f'Cound not guess noise parameter since out of {guesses}, {found} were found')
        else:
            noisepar = found[0]

    # Copy the simulations
    iterkwargs = {'ind':np.arange(n)}
    kwargs = {'sim':sim, 'noise':noise, 'noisepar':noisepar, 'verbose':verbose}
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
