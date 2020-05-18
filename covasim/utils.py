'''
Numerical utilities for running Covasim
'''

#%% Housekeeping

import numba  as nb # For faster computations
import numpy  as np # For numerics
from . import defaults as cvd

# What functions are externally visible -- note, this gets populated in each section below
__all__ = []

# Set dtypes
nbbool = nb.bool_
if cvd.default_precision == 32:
    nbint   = nb.int32
    nbfloat = nb.float32
elif cvd.default_precision == 64:
    nbint   = nb.int64
    nbfloat = nb.float64
else:
    raise NotImplementedError


#%% The core Covasim functions -- compute the infections

@nb.njit(             (nbint, nbfloat[:], nbfloat[:],     nbfloat[:], nbfloat, nbfloat, nbfloat))
def compute_viral_load(t,     time_start, time_recovered, time_dead,  frac_time,    load_ratio,    high_cap):
    '''
    Calculate relative transmissibility for time t. Includes time varying
    viral load, pre/asymptomatic factor, diagonsis factor, etc.

    Args:
        t: (int) timestep
        time_start: (float[]) individuals' infectious date
        time_recovered: (float[]) individuals' recovered date
        time_dead: (float[]) individuals' death date
        frac_time: (float) frac of time in high load
        load_ratio: (float) ratio for high to low viral load
        high_cap: (float) cap on the number of days with high viral load

    Returns:
        load (float): viral load
    '''

    # Get the end date from recover or death
    n = len(time_dead)
    time_stop = np.ones(n, dtype=cvd.default_float)*time_recovered # This is needed to make a copy
    inds = ~np.isnan(time_dead)
    time_stop[inds] = time_dead[inds]

    # Calculate which individuals with be high past the cap and when it should happen
    infect_days_total = time_stop-time_start
    trans_day = frac_time*infect_days_total
    inds = trans_day > high_cap
    cap_frac = high_cap/infect_days_total[inds]

    # Get corrected time to switch from high to low
    trans_point = np.ones(n,dtype=cvd.default_float)*frac_time
    trans_point[inds] = cap_frac

    # Calculate load
    load = np.ones(n, dtype=cvd.default_float) # allocate an array of ones with the correct dtype
    early = (t-time_start)/infect_days_total < trans_point # are we in the early or late phase
    load = (load_ratio * early + load * ~early)/(load+frac_time*(load_ratio-load)) # calculate load

    return load


@nb.njit(            (nbfloat[:], nbfloat[:], nbbool[:], nbbool[:], nbfloat,    nbfloat[:], nbbool[:], nbbool[:], nbbool[:], nbfloat,      nbfloat,    nbfloat), cache=True)
def compute_trans_sus(rel_trans,  rel_sus,    inf,       sus,       beta_layer, viral_load, symp,      diag,      quar,      asymp_factor, iso_factor, quar_factor):
    ''' Calculate relative transmissibility and susceptibility '''
    f_asymp   =  symp + ~symp * asymp_factor # Asymptomatic factor, changes e.g. [0,1] with a factor of 0.8 to [0.8,1.0]
    f_iso     = ~diag +  diag * iso_factor # Isolation factor, changes e.g. [0,1] with a factor of 0.2 to [1,0.2]
    f_quar    = ~quar +  quar * quar_factor # Quarantine, changes e.g. [0,1] with a factor of 0.5 to [1,0.5]
    rel_trans = rel_trans * inf * f_quar * f_asymp * f_iso * beta_layer * viral_load # Recalulate transmisibility
    rel_sus   = rel_sus   * sus * f_quar # Recalulate susceptibility
    return rel_trans, rel_sus


@nb.njit(             (nbfloat,  nbint[:], nbint[:],  nbfloat[:],  nbfloat[:], nbfloat[:]), cache=True)
def compute_infections(beta,     sources,  targets,   layer_betas, rel_trans,  rel_sus):
    ''' The heaviest step of the model -- figure out who gets infected on this timestep '''
    betas           = beta * layer_betas  * rel_trans[sources] * rel_sus[targets] # Calculate the raw transmission probabilities
    nonzero_inds    = betas.nonzero()[0] # Find nonzero entries
    nonzero_betas   = betas[nonzero_inds] # Remove zero entries from beta
    nonzero_sources = sources[nonzero_inds] # Remove zero entries from the sources
    nonzero_targets = targets[nonzero_inds] # Remove zero entries from the targets
    transmissions   = (np.random.random(len(nonzero_betas)) < nonzero_betas).nonzero()[0] # Compute the actual infections!
    source_inds     = nonzero_sources[transmissions]
    target_inds     = nonzero_targets[transmissions] # Filter the targets on the actual infections
    return source_inds, target_inds


#%% Sampling and seed methods

__all__ += ['sample', 'set_seed']


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

    **Examples**::

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
        if par1>0:
            mean  = np.log(par1**2 / np.sqrt(par2 + par1**2)) # Computes the mean of the underlying normal distribution
            sigma = np.sqrt(np.log(par2/par1**2 + 1)) # Computes sigma for the underlying normal distribution
            samples = np.random.lognormal(mean=mean, sigma=sigma, size=size)
        else:
            samples = np.zeros(size)
        if dist == 'lognormal_int': samples = np.round(samples)
    elif dist == 'neg_binomial':  samples = np.random.negative_binomial(n=par1, p=par2, size=size)
    else:
        choicestr = '\n'.join(choices)
        errormsg = f'The selected distribution "{dist}" is not implemented; choices are: {choicestr}'
        raise NotImplementedError(errormsg)

    return samples


def set_seed(seed=None):
    ''' Reset the random seed -- complicated because of Numba '''

    @nb.njit((nbint,), cache=True)
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


#%% Probabilities -- mostly not jitted since performance gain is minimal

__all__ += ['n_binomial', 'binomial_arr', 'binomial_filter', 'multinomial',
            'poisson', 'n_poisson', 'choose', 'choose_r', 'choose_w']

def n_binomial(prob, n):
    ''' Perform n binomial (Bernolli) trials -- return boolean array '''
    return np.random.random(n) < prob


def binomial_arr(prob_arr):
    ''' Binomial (Bernoulli) trials each with different probabilities -- return boolean array '''
    return np.random.random(len(prob_arr)) < prob_arr


def binomial_filter(prob, arr): # No speed gain from Numba
    ''' Binomial "filter" -- return entries that passed '''
    return arr[(np.random.random(len(arr)) < prob).nonzero()[0]]


def multinomial(probs, repeats): # No speed gain from Numba
    ''' A multinomial trial '''
    return np.searchsorted(np.cumsum(probs), np.random.random(repeats))


@nb.njit((nbint,), cache=True) # This hugely increases performance
def poisson(rate):
    ''' A Poisson trial '''
    return np.random.poisson(rate, 1)[0]


@nb.njit((nbint, nbint), cache=True) # This hugely increases performance
def n_poisson(rate, n):
    ''' A Poisson trial '''
    return np.random.poisson(rate, n)


@nb.njit((nbint, nbint), cache=True) # This hugely increases performance
def choose(max_n, n):
    '''
    Choose a subset of items (e.g., people) without replacement.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose

    **Example**
    ::

        choose(5, 2) will choose 2 out of 5 people with equal probability.
    '''
    return np.random.choice(max_n, n, replace=False)


@nb.njit((nbint, nbint), cache=True) # This hugely increases performance
def choose_r(max_n, n):
    '''
    Choose a subset of items (e.g., people), with replacement.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose

    **Example**::

        ``choose(5, 2)`` will choose 2 out of 5 people with equal probability.
    '''
    return np.random.choice(max_n, n, replace=True)


def choose_w(probs, n, unique=True):
    '''
    Choose n items (e.g. people), each with a probability from the distribution probs.

    Args:
        probs (array): list of probabilities, should sum to 1
        n (int): number of samples to choose
        unique (bool): whether or not to ensure unique indices

    **Example**::

        ``choose_w([0.2, 0.5, 0.1, 0.1, 0.1], 2)`` will choose 2 out of 5 people with nonequal probability.
    '''
    probs = np.array(probs)
    n_choices = len(probs)
    n_samples = int(n)
    probs_sum = probs.sum()
    if probs_sum: # Weight is nonzero, rescale
        probs = probs/probs_sum
    else: # Weights are all zero, choose uniformly
        probs = np.ones(n_choices)/n_choices
    return np.random.choice(n_choices, n_samples, p=probs, replace=not(unique))


#%% Simple array operations

__all__ += ['true',   'false',   'defined',
            'itrue',  'ifalse',  'idefined',
            'itruei', 'ifalsei', 'idefinedi']


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
