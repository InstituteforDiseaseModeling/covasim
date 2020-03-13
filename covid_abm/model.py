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


def single_run(sim):
    ''' Convenience function to perform a single simulation run '''
    sim.run()
    return sim


def multi_run(orig_sim, n=4, verbose=None):
    ''' Ditto, for multiple runs '''

    # Copy the simulations
    sims = []
    for i in range(n):
        new_sim = sc.dcp(orig_sim)
        new_sim.pars['seed'] += i # Reset the seed, otherwise no point!
        new_sim.pars['n'] = int(new_sim.pars['n']/n) # Reduce the population size accordingly
        sims.append(new_sim)

    finished_sims = sc.parallelize(single_run, iterarg=sims)

    output_sim = sc.dcp(finished_sims[0])
    output_sim.pars['parallelized'] = n # Store how this was parallelized
    output_sim.pars['n'] *= n # Restore this since used in later calculations -- a bit hacky, it's true

    for sim in finished_sims[1:]: # Skip the first one
        output_sim.people.update(sim.people)
        for key,val in sim.results.items():
            if key != 't':
                output_sim.results[key] += sim.results[key]

    return output_sim



