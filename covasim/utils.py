'''
Utilities for running the COVID-ABM
'''

import numba  as nb # For faster computations
import numpy  as np # For numerics
import pandas as pd # Used for pd.unique() (better than np.unique())
import pylab  as pl # Used by fixaxis()
import sciris as sc # Used by fixaxis()
import scipy.stats as sps # Used by poisson_test()

__all__ = ['sample', 'set_seed', 'bt', 'mt', 'pt', 'choose', 'choose_weighted', 'fixaxis', 'get_doubling_time', 'poisson_test']


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


@nb.njit((nb.float64, nb.int64))
def rbt(prob, n):
    ''' A repeated Bernoulli (binomial) trial '''
    return np.random.binomial(1, prob, n)


@nb.njit((nb.float64, nb.int64))
def mbt(prob, n):
    ''' Multiple Bernoulli (binomial) trials -- return indices that passed '''
    return list((np.random.random(n) < prob).nonzero()[0])


@nb.njit((nb.float64, nb.int64[:]))
def bf(prob, arr):
    ''' Bernoulli "filter" -- return entries that passed '''
    return list(arr[(np.random.random(len(arr)) < prob).nonzero()[0]])

@nb.njit((nb.float64[:], nb.int64))
def mt(probs, repeats):
    ''' A multinomial trial '''
    return np.searchsorted(np.cumsum(probs), np.random.random(repeats))


@nb.njit((nb.int64,))
def pt(rate):
    ''' A Poisson trial '''
    return np.random.poisson(rate, 1)[0]


@nb.njit((nb.int64, nb.int64))
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


# @nb.njit((nb.float64[:], nb.int64, nb.float64))
def choose_weighted(probs, n, overshoot=1.5, eps=1e-6, max_tries=10, normalize=False):
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

    Example:
        choose_weighted([0.2, 0.5, 0.1, 0.1, 0.1], 2) will choose 2 out of 5 people with nonequal probability.

    NB: unfortunately pd.unique() is not supported by Numba, nor is
    np.unique(return_index=True), hence why this function is not jitted.
    '''

    # Ensure it's the right type and optionally normalize
    probs = np.array(probs, dtype=np.float64)
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


def get_doubling_time(sim, series=None, interval=None, start_day=None, end_day=None, moving_window=None, exp_approx=False, max_doubling_time=100, eps=1e-3, verbose=None):
    '''
    Method to calculate doubling time
    Can be used in various ways:
        1. get_doubling_time(sim, interval=[3,30]) returns the doubling time over the given interval (single float)
        2. get_doubling_time(sim, interval=[3,30], moving_window=3) returns doubling times calculated over moving windows (array)
    Instead of an interval, can pass in the start and end days (as integers - TODO, change this to accept dates)
    Can pass in a series or the name of a result
    '''

    # Set verbose level
    if verbose is None:
        verbose = sim['verbose']

    # Validate inputs: series
    if series is None or isinstance(series, str):
        if not sim.results_ready:
            raise Exception(f"Results not ready, cannot calculate doubling time")
        else:
            if series is None or series not in sim.reskeys:
                sc.printv(f"Series not supplied or not found in results; defaulting to use cumulative exposures", 1, verbose)
                series='cum_infections'
            series = sim.results[series].values
    else:
        series = sc.promotetoarray(series)

    # Validate inputs: interval
    if interval is not None:
        if len(interval) != 2:
            sc.printv(f"Interval should be a list/array/tuple of length 2, not {len(interval)}. Resetting to length of series.", 1, verbose)
            interval = [0,len(series)]
        start_day, end_day = interval[0], interval[1]

    if len(series) < end_day:
        sc.printv(f"End day {end_day} is after the series ends ({len(series)}). Resetting to length of series.", 1, verbose)
        end_day = len(series)
    int_length = end_day - start_day

    # Deal with moving window
    if moving_window is not None:
        if not sc.isnumber(moving_window):
            sc.printv(f"Moving window should be an integer; ignoring and calculating single result", 1, verbose)
            doubling_time = get_doubling_time(sim, series=series, start_day=start_day, end_day=end_day, moving_window=None, exp_approx=exp_approx)

        else:
            if not isinstance(moving_window,int):
                sc.printv(f"Moving window should be an integer; recasting {moving_window} the nearest integer... ", 1, verbose)
                moving_window = int(moving_window)
            if moving_window < 2:
                sc.printv(f"Moving window should be greater than 1; recasting {moving_window} to 2", 1, verbose)
                moving_window = 2

            doubling_time = []
            for w in range(int_length-moving_window+1):
                this_start = start_day + w
                this_end = this_start + moving_window
                this_doubling_time = get_doubling_time(sim, series=series, start_day=this_start, end_day=this_end, exp_approx=exp_approx)
                doubling_time.append(this_doubling_time)

    # Do calculations
    else:
        if not exp_approx:
            try:
                import statsmodels.api as sm
            except ModuleNotFoundError as E:
                errormsg = f'Could not import statsmodels ({E}), falling back to exponential approximation'
                print(errormsg)
                exp_approx = True
        if exp_approx:
            if series[start_day] > 0:
                r = series[end_day] / series[start_day]
                if r > 1:
                    doubling_time = int_length * np.log(2) / np.log(r)
                    doubling_time = min(doubling_time, max_doubling_time)  # Otherwise, it's unbounded
            else:
                raise ValueError(f"Can't calculate doubling time with exponential approximation when initial value is zero.")
        else:

            if np.any(series[start_day:end_day]): # Deal with zero values if possible
                nonzero = np.nonzero(series[start_day:end_day])[0]
                if len(nonzero) >= 2:
                    exog  = sm.add_constant(np.arange(len(nonzero)))
                    endog = np.log2((series[start_day:end_day])[nonzero])
                    model = sm.OLS(endog, exog)
                    doubling_rate = model.fit().params[1]
                    if doubling_rate > eps:
                        doubling_time = 1.0 / doubling_rate
                    else:
                        doubling_time = max_doubling_time
                else:
                    raise ValueError(f"Can't calculate doubling time for series {series[start_day:end_day]}. Check whether series is growing.")
            else:
                raise ValueError(f"Can't calculate doubling time for series {series[start_day:end_day]}. Check whether series is growing.")

    return doubling_time




'''
Test for ratio of Poisson intensities in two independent samples

Author: Josef Perktold
License: BSD-3

destination statsmodels

From: https://stackoverflow.com/questions/33944914/implementation-of-e-test-for-poisson-in-python

Date: 2020feb24
'''

def poisson_test(count1, count2, exposure1=1, exposure2=1, ratio_null=1,
                      method='score', alternative='2-sided'):
    '''test for ratio of two sample Poisson intensities

    If the two Poisson rates are g1 and g2, then the Null hypothesis is

    H0: g1 / g2 = ratio_null

    against one of the following alternatives

    H1_2-sided: g1 / g2 != ratio_null
    H1_larger: g1 / g2 > ratio_null
    H1_smaller: g1 / g2 < ratio_null

    Parameters
    ----------
    count1: int
        Number of events in first sample
    exposure1: float
        Total exposure (time * subjects) in first sample
    count2: int
        Number of events in first sample
    exposure2: float
        Total exposure (time * subjects) in first sample
    ratio: float
        ratio of the two Poisson rates under the Null hypothesis. Default is 1.
    method: string
        Method for the test statistic and the p-value. Defaults to `'score'`.
        Current Methods are based on Gu et. al 2008
        Implemented are 'wald', 'score' and 'sqrt' based asymptotic normal
        distribution, and the exact conditional test 'exact-cond', and its mid-point
        version 'cond-midp', see Notes
    alternative : string
        The alternative hypothesis, H1, has to be one of the following

           'two-sided': H1: ratio of rates is not equal to ratio_null (default)
           'larger' :   H1: ratio of rates is larger than ratio_null
           'smaller' :  H1: ratio of rates is smaller than ratio_null

    Returns
    -------
    pvalue two-sided # stat

    not yet
    #results : Results instance
    #    The resulting test statistics and p-values are available as attributes.


    Notes
    -----
    'wald': method W1A, wald test, variance based on separate estimates
    'score': method W2A, score test, variance based on estimate under Null
    'wald-log': W3A
    'score-log' W4A
    'sqrt': W5A, based on variance stabilizing square root transformation
    'exact-cond': exact conditional test based on binomial distribution
    'cond-midp': midpoint-pvalue of exact conditional test

    The latter two are only verified for one-sided example.

    References
    ----------
    Gu, Ng, Tang, Schucany 2008: Testing the Ratio of Two Poisson Rates,
    Biometrical Journal 50 (2008) 2, 2008

    '''

    # Copied from statsmodels.stats.weightstats
    def zstat_generic2(value, std_diff, alternative):
        '''generic (normal) z-test to save typing

        can be used as ztest based on summary statistics
        '''
        zstat = value / std_diff
        if alternative in ['two-sided', '2-sided', '2s']:
            pvalue = sps.norm.sf(np.abs(zstat))*2
        elif alternative in ['larger', 'l']:
            pvalue = sps.norm.sf(zstat)
        elif alternative in ['smaller', 's']:
            pvalue = sps.norm.cdf(zstat)
        else:
            raise ValueError('invalid alternative')
        return pvalue# zstat

    # shortcut names
    y1, n1, y2, n2 = count1, exposure1, count2, exposure2
    d = n2 / n1
    r = ratio_null
    r_d = r / d

    if method in ['score']:
        stat = (y1 - y2 * r_d) / np.sqrt((y1 + y2) * r_d)
        dist = 'normal'
    elif method in ['wald']:
        stat = (y1 - y2 * r_d) / np.sqrt(y1 + y2 * r_d**2)
        dist = 'normal'
    elif method in ['sqrt']:
        stat = 2 * (np.sqrt(y1 + 3 / 8.) - np.sqrt((y2 + 3 / 8.) * r_d))
        stat /= np.sqrt(1 + r_d)
        dist = 'normal'
    elif method in ['exact-cond', 'cond-midp']:
        from statsmodels.stats import proportion
        bp = r_d / (1 + r_d)
        y_total = y1 + y2
        stat = None
        pvalue = proportion.binom_test(y1, y_total, prop=bp, alternative=alternative)
        if method in ['cond-midp']:
            # not inplace in case we still want binom pvalue
            pvalue = pvalue - 0.5 * sps.binom.pmf(y1, y_total, bp)

        dist = 'binomial'

    if dist == 'normal':
        return zstat_generic2(stat, 1, alternative)
    else:
        return pvalue#, stat