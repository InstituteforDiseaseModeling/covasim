'''
Utilities for running the COVID-ABM
'''

import numpy as np # Needed for a few things not provided by pl
import numba as nb # For faster computations
import pylab as pl
import pandas as pd
import sciris as sc

__all__ = ['bt', 'mt', 'set_seed', 'fixaxis']

#%% Define helper functions

def set_seed(seed=None):
    ''' Reset the random seed -- complicated because of Numba '''
    
    @nb.njit((nb.int64,))
    def set_seed_numba(seed):
        return np.random.seed(seed)
    
    def set_seed_regular(seed):
        return np.random.seed(seed)
    
    if seed is not None:
        set_seed_numba(seed)
        set_seed_regular(seed)
    return


@nb.njit((nb.float64,)) # These types can also be declared as a dict, but performance is much slower...?
def bt(prob):
    ''' A simple Bernoulli (binomial) trial '''
    return np.random.random() < prob # Or rnd.random() < prob, np.random.binomial(1, prob), which seems slower

@nb.njit((nb.float64[:], nb.int64))
def mt(probs, repeats):
    ''' A multinomial trial '''
    return np.searchsorted(np.cumsum(probs), np.random.random(repeats))


@nb.njit((nb.int64, nb.int64))
def choose_people(max_ind, n):
    '''
    Choose n people.
    
    choose_people(5, 2) will choose 2 out of 5 people with equal probability.
    '''
    n_samples = min(int(n), max_ind)
    inds = pl.choice(max_ind, n_samples, replace=False)
    return inds


# @nb.njit((nb.float64[:], nb.int64, nb.float64))
def choose_people_weighted(probs, n, overshoot=1.5):
    '''
    Choose n people, each with a probability from the distribution probs. Overshoot
    handles the case where there are repeats
    
    choose_people([0.2, 0.5, 0.1, 0.1, 0.1], 2) will choose 2 out of 5 people with nonequal probability.
    
    NB: unfortunately pd.unique() is not supported by Numba, nor is
    np.unique(return_index=True), hence why this function is not jitted.
    '''
    if len(probs)>=n: # Otherwise, it's everyone
        print('Warning: number of samples requested is greater than the number of people')
    unique_inds = pl.array([nb.int64(x) for x in range(0)])
    while len(unique_inds)<n:
        raw_inds = mt(probs, n*overshoot) # Return raw indices, with replacement
        mixed_inds = pl.hstack((unique_inds, raw_inds))
        unique_inds = pd.unique(mixed_inds) # Or np.unique(mixed_inds, return_index=True) with another step
    inds = unique_inds[:n]
    return inds


def fixaxis(useSI=True):
    ''' Make the plotting more consistent -- add a legend and ensure the axes start at 0 '''
    pl.legend() # Add legend
    sc.setylim() # Rescale y to start at 0
    sc.setxlim()
    return