'''
Utilities for running the COVID-ABM
'''

import numba  as nb # For faster computations
import numpy  as np # For numerics
import pandas as pd # Used for pd.unique() (better than np.unique())
import pylab  as pl # Used by fixaxis()
import sciris as sc # Used by fixaxis()

__all__ = ['set_seed', 'bt', 'mt', 'pt', 'choose_people', 'choose_people_weighted', 'fixaxis']

#%% Define helper functions

def set_seed(seed=None):
    ''' Reset the random seed -- complicated because of Numba '''

    @nb.njit((nb.int64,))
    def set_seed_numba(seed):
        return np.random.seed(seed)

    def set_seed_regular(seed):
        return np.random.seed(seed)

    set_seed_regular(seed)
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
        raise Exception('Number of samples requested is greater than the number of people')
    n_samples = min(int(n), max_ind)
    inds = np.random.choice(max_ind, n_samples, replace=False)
    return inds


# @nb.njit((nb.float64[:], nb.int64, nb.float64))
def choose_people_weighted(probs, n, overshoot=1.5, eps=1e-6):
    '''
    Choose n people, each with a probability from the distribution probs. Overshoot
    handles the case where there are repeats

    choose_people([0.2, 0.5, 0.1, 0.1, 0.1], 2) will choose 2 out of 5 people with nonequal probability.

    NB: unfortunately pd.unique() is not supported by Numba, nor is
    np.unique(return_index=True), hence why this function is not jitted.
    '''
    probs = np.array(probs, dtype=np.float64)
    n = int(n)
    if abs(probs.sum() - 1) > eps:
        raise Exception('Probabilities should sum to 1')
    if len(probs) < n: # Otherwise, it's everyone
        raise Exception('Number of samples requested is greater than the number of people')
    unique_inds = np.array([], dtype=np.int)
    while len(unique_inds)<n:
        raw_inds = mt(probs, int(n*overshoot)) # Return raw indices, with replacement
        mixed_inds = np.hstack((unique_inds, raw_inds))
        unique_inds = pd.unique(mixed_inds) # Or np.unique(mixed_inds, return_index=True) with another step
    inds = unique_inds[:n]
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
