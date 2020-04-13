'''
Numerical utilities for running Covasim
'''

import numba  as nb # For faster computations
import numpy  as np # For numerics
import pandas as pd # Used for pd.unique() (better than np.unique())

# __all__ = ['sample', 'set_seed', 'binomial', 'multinomial', 'poisson', 'choose', 'choose_weighted']


def sample(dist=None, par1=None, par2=None, size=None):
    '''
    Draw a sample from the distribution specified by the input.

    Args:
        dist (str): the distribution to sample from
        par1 (float): the "main" distribution parameter (e.g. mean)
        par2 (float): the "secondary" distribution parameter (e.g. std)
        size (int): the number of samples (default=1)

    Returns:
        A length N array of samples

    Examples:
        sample() # returns Unif(0,1)
        sample(dist='normal', par1=3, par2=0.5) # returns Normal(μ=3, σ=0.5)

    Notes:
        Lognormal distributions are parameterized with reference to the underlying normal distribution (see:
        https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.lognormal.html), but this function assumes
        the user wants to specify the mean and variance of the lognormal distribution


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

    # Compute distribution parameters and draw samples
    # NB, if adding a new distribution, also add to choices above
    if   dist == 'uniform':       samples = np.random.uniform(low=par1, high=par2, size=size)
    elif dist == 'normal':        samples = np.random.normal(loc=par1, scale=par2, size=size)
    elif dist == 'normal_pos':    samples = np.abs(np.random.normal(loc=par1, scale=par2, size=size))
    elif dist == 'normal_int':    samples = np.round(np.abs(np.random.normal(loc=par1, scale=par2, size=size)))
    elif dist in ['lognormal', 'lognormal_int']:
        mean  = np.log(par1**2 / np.sqrt(par2 + par1**2)) # Computes the mean of the underlying normal distribution
        sigma = np.sqrt(np.log(par2/par1**2 + 1)) # Computes sigma for the underlying normal distribution
        samples = np.random.lognormal(mean=mean, sigma=sigma, size=size)
        if dist == 'lognormal_int': samples = np.round(samples)
    elif dist == 'neg_binomial':  samples = np.random.negative_binomial(n=par1, p=par2, size=size)
    else:
        choicestr = '\n'.join(choices)
        errormsg = f'The selected distribution "{dist}" is not implemented; choices are: {choicestr}'
        raise NotImplementedError(errormsg)

    return samples


def set_seed(seed=None):
    ''' Reset the random seed -- complicated because of Numba '''

    @nb.njit((nb.int32,))
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


# @nb.njit((nb.boolean[:],))
def true(arr):
    ''' Retrurns the indices of the values of the array that are true '''
    return arr.nonzero()[0]

# @nb.njit((nb.boolean[:],))
def false(arr):
    ''' Retrurns the indices of the values of the array that are false '''
    return (~arr).nonzero()[0]

# @nb.njit((nb.float32[:],))
def defined(arr):
    ''' Retrurns the indices of the values of the array that are not-nan '''
    return (~np.isnan(arr)).nonzero()[0]

def itrue(arr, inds):
    ''' Retrurns the indices of the values of the array that are true '''
    return inds[arr[inds]]

def ifalse(arr, inds):
    ''' Retrurns the indices of the values of the array that are false '''
    return inds[~arr[inds]]

def idefined(arr, inds):
    ''' Retrurns the indices of the values of the array that are not-nan '''
    return inds[~np.isnan(arr[inds])]

def true_inds(arr, inds):
    ''' Retrurns the indices of the values of the array that are true '''
    return inds[arr]

def false_inds(arr, inds):
    ''' Retrurns the indices of the values of the array that are false '''
    return inds[~arr]

def defined_inds(arr, inds):
    ''' Retrurns the indices of the values of the array that are not-nan '''
    return inds[~np.isnan(arr)]

# @nb.njit((nb.float32[:],))
def binomial_arr(prob_arr):
    ''' Bernoulli trial array -- return boolean '''
    return np.random.random(len(prob_arr)) < prob_arr

# @nb.njit((nb.float32, nb.int32))
def repeated_binomial(prob, n):
    ''' A repeated Bernoulli (binomial) trial '''
    return np.random.binomial(1, prob, n)


# @nb.njit((nb.float32[:], nb.int32))
def multinomial(probs, repeats):
    ''' A multinomial trial '''
    return np.searchsorted(np.cumsum(probs), np.random.random(repeats))


@nb.njit((nb.int32,)) # This hugely increases performance
def poisson(rate):
    ''' A Poisson trial '''
    return np.random.poisson(rate, 1)[0]


@nb.njit((nb.int32, nb.int32)) # This hugely increases performance
def choose(max_n, n):
    '''
    Choose a subset of items (e.g., people) without replace.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose

    Example:
        choose(5, 2) will choose 2 out of 5 people with equal probability.
    '''
    return np.random.choice(max_n, n, replace=False)


def choose_weighted(probs, n, overshoot=1.5, eps=1e-6, max_tries=10, normalize=False, unique=True):
    '''
    Choose n items (e.g. people), each with a probability from the distribution probs.
    Overshoot handles the case where there are repeats.

    Args:
        probs (array): list of probabilities, should sum to 1
        n (int): number of samples to choose
        overshoot (float): number of extra samples to generate, expecting duplicates
        eps (float): how close to check that probabilities sum to 1
        max_tries (int): maximum number of times to try to pick samples without replacement
        normalize (bool): whether or not to normalize probs to always sum to 1
        unique (bool): whether or not to ensure unique indices

    Example:
        choose_weighted([0.2, 0.5, 0.1, 0.1, 0.1], 2) will choose 2 out of 5 people with nonequal probability.

    NB: unfortunately pd.unique() is not supported by Numba, nor is
    np.unique(return_index=True), hence why this function is not jitted.
    '''

    # Ensure it's the right type and optionally normalize
    if not unique:
        overshoot = 1
    probs = np.array(probs, dtype=np.float32)
    n_people = len(probs)
    n_samples = int(n)
    if normalize:
        probs_sum = probs.sum()
        if probs_sum: # Weight is nonzero, rescale
            probs /= probs_sum
        else: # Weights are all zero, choose uniformly
            probs = np.ones(n_people)/n_people

    # Perform checks
    if abs(probs.sum() - 1) > eps:
        raise Exception('Probabilities should sum to 1')
    if n_people == n_samples: # It's everyone
        return np.arange(len(probs))
    if n_people < n_samples: # It's more than everyone
        errormsg = f'Number of samples requested ({n_samples}) is greater than the number of people ({n_people})'
        raise Exception(errormsg)

    # Choose samples
    unique_inds = np.array([], dtype=np.int)
    tries = 0
    while len(unique_inds)<n_samples and tries<max_tries:
        tries += 1
        raw_inds = multinomial(probs, int(n_samples*overshoot)) # Return raw indices, with replacement
        if not unique:
            return raw_inds
        mixed_inds = np.hstack((unique_inds, raw_inds))
        unique_inds = pd.unique(mixed_inds) # Or np.unique(mixed_inds, return_index=True) with another step
    if tries == max_tries:
        errormsg = f'Unable to choose {n_samples} unique samples from {n_people} people after {max_tries} tries'
        raise RuntimeError(errormsg)
    inds = unique_inds[:int(n)]

    return inds


@nb.njit((nb.float32, nb.int32[:], nb.int32[:], nb.float32[:], nb.float32[:], nb.float32[:]))
def compute_targets(beta, sources, targets, layer_betas, rel_trans, rel_sus):
    ''' The heaviest step of the model -- figure out who gets infected on this timestep '''
    betas   = beta * layer_betas  * rel_trans[sources] * rel_sus[targets]
    nonzero_inds = betas.nonzero()[0]
    nonzero_betas = betas[nonzero_inds]
    nonzero_targets = targets[nonzero_inds]
    transmissions = (np.random.random(len(nonzero_betas)) < nonzero_betas).nonzero()[0]
    transmission_inds = nonzero_targets[transmissions]
    return transmission_inds