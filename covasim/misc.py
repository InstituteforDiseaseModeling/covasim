'''
Miscellaneous functions that do not belong anywhere else
'''

import numpy as np
import pylab  as pl # Used by fixaxis()
import sciris as sc # Used by fixaxis()
import scipy.stats as sps # Used by poisson_test()
from . import version as cvver


__all__ = ['check_version', 'git_info', 'fixaxis', 'progressbar', 'get_doubling_time', 'poisson_test']


def check_version(expected, die=False, verbose=True, **kwargs):
    '''
    Get current git information and optionally write it to disk.

    Args:
        expected (str): expected version information
        die (bool): whether or not to raise an exception if the check fails
    '''
    version = cvver.__version__
    compare = sc.compareversions(version, expected) # Returns -1, 0, or 1
    relation = ['older', '', 'newer'][compare+1] # Picks the right string
    if relation: # Not empty, print warning
        string = f'Note: Covasim is {relation} than expected ({version} vs. {expected})'
        if die:
            raise ValueError(string)
        elif verbose:
            print(string)
    return compare


def git_info(filename=None, check=False, old_info=None, die=False, verbose=True, **kwargs):
    '''
    Get current git information and optionally write it to disk.

    Args:
        filename (str): name of the file to write to or read from
        check (bool): whether or not to compare two git versions
        old_info (dict): dictionary of information to check against
        die (bool): whether or not to raise an exception if the check fails

    Example:
        cv.git_info('covasim_version.json') # Writes to disk
        cv.git_info('covasim_version.json', check=True) # Checks that current version matches saved file
    '''
    info = sc.gitinfo(__file__)
    if not check: # Just get information
        if filename is not None:
            output = sc.savejson(filename, info, **kwargs)
        else:
            output = info
    else:
        if filename is not None:
            old_info = sc.loadjson(filename, **kwargs)
        string = ''
        if info != old_info:
            string = f'Git information differs: {info} vs. {old_info}'
            if die:
                raise ValueError(string)
            elif verbose:
                print(string)
    return output


def fixaxis(sim, useSI=True, boxoff=False):
    ''' Make the plotting more consistent -- add a legend and ensure the axes start at 0 '''
    delta = 0.5
    pl.legend() # Add legend
    sc.setylim() # Rescale y to start at 0
    pl.xlim((0, sim['n_days']+delta))
    if boxoff:
        sc.boxoff() # Turn off top and right lines
    return


def progressbar(i, maxiters, label='', length=30, empty='—', full='•', newline=False):
    '''
    Call in a loop to create terminal progress bar.

    Args:
        i (int): current iteration
        maxiters (int): maximum number of iterations
        label (str): initial label to print
        length (int): length of progress bar
        empty (str): character for empty steps
        full (str): character for empty steps

    **Example**
    ::

        import pylab as pl
        for i in range(100):
            progressbar(i+1, 100)
            pl.pause(0.05)

    Adapted from example by Greenstick (https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console)
    '''
    ending = None if newline else '\r'
    pct = i/maxiters*100
    percent = f'{pct:0.0f}%'
    filled = int(length*i//maxiters)
    bar = full*filled + empty*(length-filled)
    print(f'\r{label} {bar} {percent}', end=ending)
    if i == maxiters: print()
    return


def get_doubling_time(sim, series=None, interval=None, start_day=None, end_day=None, moving_window=None, exp_approx=False, max_doubling_time=100, eps=1e-3, verbose=None):
    '''
    Method to calculate doubling time.

    **Examples**
    ::

        get_doubling_time(sim, interval=[3,30]) # returns the doubling time over the given interval (single float)
        get_doubling_time(sim, interval=[3,30], moving_window=3) # returns doubling times calculated over moving windows (array)
    '''
    
    # Set verbose level
    if verbose is None:
        verbose = sim['verbose']

    # Validate inputs: series
    if series is None or isinstance(series, str):
        if not sim.results_ready:
            raise Exception(f"Results not ready, cannot calculate doubling time")
        else:
            if series is None or series not in sim.result_keys():
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
