'''
Utilities for running the COVID-ABM
'''

import numba  as nb # For faster computations
import numpy  as np # For numerics
import pandas as pd # Used for pd.unique() (better than np.unique())
import pylab  as pl # Used by fixaxis()
import sciris as sc # Used by fixaxis()

__all__ = ['sample', 'set_seed', 'bt', 'mt', 'pt', 'choose_people', 'choose_people_weighted', 'fixaxis']

#%% Define helper functions

def sample(dist=None, par1=None, par2=None, size=None):
    '''
    Draw a sample from the distribution specified by the input.

    Args:
        dist (str): the distribution to sample from
        par1 (float): the "main" distribution parameter (e.g. mean)
        par2 (float): the "secondary" distribution parameter (e.g. std)
        n (int): the number of samples (default=1)

    Returns:
        A length N array of samples

    Examples:
        sample() # returns Unif(0,1)
        sample(dist='normal', par1=3, par2=0.5) # returns Normal(μ=3, σ=0.5)

    '''

    choices = [
        'uniform',
        'normal',
        'lognormal',
        'normal_pos',
        'normal_int',
        'lognormal_int',
        'neg_binomial'
        ]

    # NB, if adding a new distribution, also add to choices above
    if   dist == 'uniform':       samples = np.random.uniform(low=par1, high=par2, size=size)
    elif dist == 'normal':        samples = np.random.normal(loc=par1, scale=par2, size=size)
    elif dist == 'normal_pos':    samples = np.abs(np.random.normal(loc=par1, scale=par2, size=size))
    elif dist == 'normal_int':    samples = np.round(np.abs(np.random.normal(loc=par1, scale=par2, size=size)))
    elif dist == 'lognormal':     samples = np.random.lognormal(mean=par1, sigma=par2, size=size)
    elif dist == 'lognormal_int': samples = np.round(np.random.lognormal(mean=par1, sigma=par2, size=size))
    elif dist == 'neg_binomial':  samples = np.random.negative_binomial(n=par1, p=par2, size=size)
    else:
        choicestr = '\n'.join(choices)
        errormsg = f'The selected distribution "{dist}" is not implemented; choices are: {choicestr}'
        raise NotImplementedError(errormsg)

    return samples


def set_seed(seed=None):
    ''' Reset the random seed -- complicated because of Numba '''

    @nb.njit((nb.int64,))
    def set_seed_numba(seed):
        return np.random.seed(seed)

    def set_seed_regular(seed):
        return np.random.seed(seed)

    # Dies if a float is given
    if seed is not None:
        seed = int(seed)

    set_seed_regular(seed) # If None, reinitializes it
    if seed is None: # Numba can't accept a None seed, so use our just-reinitialized Numpy stream to generate one
        seed = np.random.randint(1e9)
    set_seed_numba(seed)

    return


@nb.njit((nb.float64,)) # These types can also be declared as a dict, but performance is much slower...?
def bt(prob):
    ''' A simple Bernoulli (binomial) trial '''
    return np.random.random() < prob # Or rnd.random() < prob, np.random.binomial(1, prob), which seems slower


@nb.njit((nb.float64[:], nb.int64))
def mt(probs, repeats):
    ''' A multinomial trial '''
    return np.searchsorted(np.cumsum(probs), np.random.random(repeats))


@nb.njit((nb.int64,))
def pt(rate):
    ''' A Poisson trial '''
    return np.random.poisson(rate, 1)[0]


@nb.njit((nb.int64, nb.int64))
def choose_people(max_ind, n):
    '''
    Choose n people.

    choose_people(5, 2) will choose 2 out of 5 people with equal probability.
    '''
    if max_ind < n:
        raise Exception('Number of samples requested is greater than the number of people') # NB: because it's Numba, can't display values
    n_samples = min(int(n), max_ind)
    inds = np.random.choice(max_ind, n_samples, replace=False)
    return inds


# @nb.njit((nb.float64[:], nb.int64, nb.float64))
def choose_people_weighted(probs, n, overshoot=1.5, eps=1e-6, max_tries=10):
    '''
    Choose n people, each with a probability from the distribution probs. Overshoot
    handles the case where there are repeats

    choose_people([0.2, 0.5, 0.1, 0.1, 0.1], 2) will choose 2 out of 5 people with nonequal probability.

    NB: unfortunately pd.unique() is not supported by Numba, nor is
    np.unique(return_index=True), hence why this function is not jitted.
    '''
    probs = np.array(probs, dtype=np.float64)
    n_samples = int(n)
    n_people = len(probs)
    if abs(probs.sum() - 1) > eps:
        raise Exception('Probabilities should sum to 1')
    if n_people == n_samples: # It's everyone
        return np.arange(len(probs))
    if n_people < n_samples: # It's more than everyone
        errormsg = f'Number of samples requested ({n_samples}) is greater than the number of people ({n_people})'
        raise Exception(errormsg)
    unique_inds = np.array([], dtype=np.int)
    tries = 0
    while len(unique_inds)<n_samples and tries<max_tries:
        tries += 1
        raw_inds = mt(probs, int(n_samples*overshoot)) # Return raw indices, with replacement
        mixed_inds = np.hstack((unique_inds, raw_inds))
        unique_inds = pd.unique(mixed_inds) # Or np.unique(mixed_inds, return_index=True) with another step
    if tries == max_tries:
        errormsg = f'Unable to choose {n_samples} unique samples from {n_people} people after {max_tries} tries'
        raise RuntimeError(errormsg)
    inds = unique_inds[:int(n)]
    return inds


def fixaxis(sim, useSI=True, boxoff=False):
    ''' Make the plotting more consistent -- add a legend and ensure the axes start at 0 '''
    delta = 0.5
    pl.legend() # Add legend
    sc.setylim() # Rescale y to start at 0
    pl.xlim((0, sim['n_days']+delta))
    if boxoff:
        sc.boxoff() # Turn off top and right lines
    return