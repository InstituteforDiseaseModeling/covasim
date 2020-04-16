'''
Numerical utilities for running Covasim
'''

import numba  as nb # For faster computations
import numpy  as np # For numerics
import pandas as pd # Used for pd.unique() (better than np.unique())


#%% Sampling and seed methods

__all__ = ['sample', 'set_seed']


def sample(dist=None, par1=None, par2=None, size=None):
    '''
    Draw a sample from the distribution specified by the input.

    Args:
        dist (str):   the distribution to sample from
        par1 (float): the "main" distribution parameter (e.g. mean)
        par2 (float): the "secondary" distribution parameter (e.g. std)
        size (int):   the number of samples (default=1)

    Returns:
        A length N array of samples

    Examples:
        sample() # returns Unif(0,1)
        sample(dist='normal', par1=3, par2=0.5) # returns Normal(μ=3, σ=0.5)

    Notes:
        Lognormal distributions are parameterized with reference to the underlying normal distribution (see:
        https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.lognormal.html), but this
        function assumes the user wants to specify the mean and variance of the lognormal distribution.
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


#%% Simple array operations
__all__ += ['true', 'false', 'defined',
            'itrue', 'ifalse', 'idefined',
            'itruei', 'ifalsei', 'idefinedi',
            ]


def true(arr):
    ''' Returns the indices of the values of the array that are true '''
    return arr.nonzero()[0]

def false(arr):
    ''' Returns the indices of the values of the array that are false '''
    return (~arr).nonzero()[0]

def defined(arr):
    ''' Returns the indices of the values of the array that are not-nan '''
    return (~np.isnan(arr)).nonzero()[0]

def itrue(arr, inds):
    ''' Returns the indices that are true in the array -- name is short for indices[true] '''
    return inds[arr]

def ifalse(arr, inds):
    ''' Returns the indices that are true in the array -- name is short for indices[false] '''
    return inds[~arr]

def idefined(arr, inds):
    ''' Returns the indices that are true in the array -- name is short for indices[defined] '''
    return inds[~np.isnan(arr)]

def itruei(arr, inds):
    ''' Returns the indices that are true in the array -- name is short for indices[true[indices]] '''
    return inds[arr[inds]]

def ifalsei(arr, inds):
    ''' Returns the indices that are false in the array -- name is short for indices[false[indices]] '''
    return inds[~arr[inds]]

def idefinedi(arr, inds):
    ''' Returns the indices that are defined in the array -- name is short for indices[defined[indices]] '''
    return inds[~np.isnan(arr[inds])]



#%% Probabilities -- mostly not jitted since performance gain is minimal

__all__ += ['binomial_arr', 'multinomial', 'poisson', 'binomial_filter', 'choose', 'choose_r', 'choose_w']


def n_binomial(prob, n):
    ''' Perform n binomial (Bernolli) trials -- return boolean array '''
    return np.random.random(n) < prob

def binomial_arr(prob_arr):
    ''' Binomial (Bernoulli) trials each with different probabilities -- return boolean array '''
    return np.random.random(len(prob_arr)) < prob_arr


def multinomial(probs, repeats):
    ''' A multinomial trial '''
    return np.searchsorted(np.cumsum(probs), np.random.random(repeats))


@nb.njit((nb.int32,)) # This hugely increases performance
def poisson(rate):
    ''' A Poisson trial '''
    return np.random.poisson(rate, 1)[0]


#@nb.njit((nb.float64, nb.int64[:]))
def binomial_filter(prob, arr):
    ''' Binomial "filter" -- return entries that passed '''
    return arr[(np.random.random(len(arr)) < prob).nonzero()[0]]


@nb.njit((nb.int32, nb.int32)) # This hugely increases performance
def choose(max_n, n):
    '''
    Choose a subset of items (e.g., people) without replacement.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose

    Example:
        choose(5, 2) will choose 2 out of 5 people with equal probability.
    '''
    return np.random.choice(max_n, n, replace=False)


@nb.njit((nb.int32, nb.int32)) # This hugely increases performance
def choose_r(max_n, n):
    '''
    Choose a subset of items (e.g., people), with replacement.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose

    Example:
        choose(5, 2) will choose 2 out of 5 people with equal probability.
    '''
    return np.random.choice(max_n, n, replace=True)


def choose_w(probs, n, unique=True):
    '''
    Choose n items (e.g. people), each with a probability from the distribution probs.

    Args:
        probs (array): list of probabilities, should sum to 1
        n (int): number of samples to choose
        unique (bool): whether or not to ensure unique indices

    Example:
        choose_w([0.2, 0.5, 0.1, 0.1, 0.1], 2) will choose 2 out of 5 people with nonequal probability.
    '''
    probs = np.array(probs, dtype=np.float32)
    n_choices = len(probs)
    n_samples = int(n)
    probs_sum = probs.sum()
    if probs_sum: # Weight is nonzero, rescale
        probs /= probs_sum
    else: # Weights are all zero, choose uniformly
        probs = np.ones(n_choices)/n_choices
    return np.random.choice(n_choices, n_samples, p=probs, replace=not(unique))


#%% The core Covasim functions -- compute the infections

@nb.njit((    nb.float32[:], nb.float32[:], nb.bool_[:], nb.bool_[:], nb.bool_[:],   nb.float32,  nb.float32, nb.float32))
def compute_probs(rel_trans,       rel_sus,        symp,        diag,        quar, asymp_factor, diag_factor, quar_trans):
    ''' Calculate relative transmissibility and susceptibility '''
    f_asymp    =  symp + ~symp * asymp_factor # Asymptomatic factor, changes e.g. [0,1] with a factor of 0.8 to [0.8,1.0]
    f_diag     = ~diag +  diag * diag_factor # Diagnosis factor, changes e.g. [0,1] with a factor of 0.8 to [1,0.8]
    f_quar_eff = ~quar +  quar * quar_trans # Quarantine
    rel_trans  = rel_trans * f_quar_eff * f_asymp * f_diag # Recalulate transmisibility
    rel_sus    = rel_sus   * f_quar_eff # Recalulate susceptibility
    return rel_trans, rel_sus


@nb.njit((    nb.float32, nb.int32[:], nb.int32[:], nb.float32[:], nb.float32[:], nb.float32[:]))
def compute_targets(beta,     sources,     targets,   layer_betas,     rel_trans,       rel_sus):
    ''' The heaviest step of the model -- figure out who gets infected on this timestep '''
    betas           = beta * layer_betas  * rel_trans[sources] * rel_sus[targets] # Calculate the raw transmission probabilities
    nonzero_inds    = betas.nonzero()[0] # Find nonzero entries
    nonzero_betas   = betas[nonzero_inds] # Remove zero entries from beta
    nonzero_targets = targets[nonzero_inds] # Remove zero entries from the targets
    transmissions   = (np.random.random(len(nonzero_betas)) < nonzero_betas).nonzero()[0] # Compute the actual infections!
    edge_inds       = nonzero_inds[transmissions] # The index of the contact responsible for the transmission
    target_inds     = nonzero_targets[transmissions] # Filter the targets on the actual infections
    target_inds     = np.unique(target_inds) # Ensure the targets are unique
    return target_inds, edge_inds