'''
Miscellaneous functions that do not belong anywhere else
'''

import numpy as np
import pandas as pd
import sciris as sc
import datetime as dt
import scipy.stats as sps
from . import version as cvver


__all__ = ['load_data', 'date', 'daydiff', 'load', 'save', 'check_version', 'git_info', 'get_doubling_time', 'poisson_test']


def load_data(filename, columns=None, calculate=True, verbose=True, **kwargs):
    '''
    Load data for comparing to the model output.

    Args:
        filename (str): the name of the file to load (either Excel or CSV)
        columns (list): list of column names (otherwise, load all)
        calculate (bool): whether or not to calculate cumulative values from daily counts
        kwargs (dict): passed to pd.read_excel()

    Returns:
        data (dataframe): pandas dataframe of the loaded data
    '''

    # Load data
    if filename.lower().endswith('csv'):
        raw_data = pd.read_csv(filename, **kwargs)
    elif filename.lower().endswith('xlsx'):
        raw_data = pd.read_excel(filename, **kwargs)
    else:
        errormsg = f'Currently loading is only supported from .csv and .xlsx files, not {filename}'
        raise NotImplementedError(errormsg)

    # Confirm data integrity and simplify
    if columns is not None:
        for col in columns:
            if col not in raw_data.columns:
                errormsg = f'Column "{col}" is missing from the loaded data'
                raise ValueError(errormsg)
        data = raw_data[columns]
    else:
        data = raw_data

    # Calculate any cumulative columns that are missing
    if calculate:
        columns = data.columns
        for col in columns:
            if col.startswith('new'):
                cum_col = col.replace('new_', 'cum_')
                if cum_col not in columns:
                    data[cum_col] = np.cumsum(data[col])
                    if verbose:
                        print(f'  Automatically adding cumulative column {cum_col} from {col}')

    # Ensure required columns are present
    if 'date' not in data.columns:
        errormsg = f'Required column "date" not found; columns are {data.columns}'
        raise ValueError(errormsg)
    else:
        data['date'] = pd.to_datetime(data['date']).dt.date

    data.set_index('date', inplace=True, drop=False) # So sim.data['date'] can still be accessed

    return data


def date(obj, *args, **kwargs):
    '''
    Convert a string or a datetime object to a date object. To convert to an integer
    from the start day, use sim.date() instead.

    Args:
        obj (str, date, datetime): the object to convert
        args (str, date, datetime): additional objects to convert

    Returns:
        dates (date or list): either a single date object, or a list of them

    **Examples**::

        cv.date('2020-04-05') # Returns datetime.date(2020, 4, 5)
    '''
    # Convert to list
    obj = sc.promotetolist(obj) # Ensure it's iterable
    obj.extend(args)

    dates = []
    for d in obj:
        try:
            if type(d) == dt.date: # Do not use isinstance, since must be the exact type
                pass
            elif sc.isstring(d):
                d = sc.readdate(d).date()
            elif isinstance(d, dt.datetime):
                d = d.date()
            else:
                errormsg = f'Cannot interpret {type(d)} as a date, must be date, datetime, or string'
                raise TypeError(errormsg)
            dates.append(d)
        except Exception as E:
            errormsg = f'Conversion of "{d}" to a date failed: {str(E)}'
            raise ValueError(errormsg)

    # Return an integer rather than a list if only one provided
    if len(dates)==1:
        dates = dates[0]

    return dates


def daydiff(*args):
    '''
    Convenience function to find the difference between two or more days. With
    only one argument, calculate days sin 2020-01-01.

    **Example**::

        since_ny = cv.daydiff('2020-03-20') # Returns 79 days since Jan. 1st
        diff     = cv.daydiff('2020-03-20', '2020-04-05') # Returns 16
        diffs    = cv.daydiff('2020-03-20', '2020-04-05', '2020-05-01') # Returns [16, 26]
    '''
    days = [date(day) for day in args]
    if len(days) == 1:
        days.insert(0, date('2020-01-01')) # With one date, return days since Jan. 1st

    output = []
    for i in range(len(days)-1):
        diff = (days[i+1] - days[i]).days
        output.append(diff)

    if len(output) == 1:
        output = output[0]

    return output


def load(*args, **kwargs):
    '''
    Convenience method for sc.loadobj() and equivalent to cv.Sim.load() or
    cv.Scenarios.load().

    **Examples**::

        sim = cv.load('calib.sim')
        scens = cv.load(filename='school-closures.scens', folder='schools')
    '''
    obj = sc.loadobj(*args, **kwargs)
    if hasattr(obj, 'version'):
        v_curr = cvver.__version__
        v_obj = obj.version
        cmp = check_version(v_obj, verbose=False)
        if cmp != 0:
            print(f'Note: you have Covasim v{v_curr}, but are loading an object from v{v_obj}')
    return obj


def save(*args, **kwargs):
    '''
    Convenience method for sc.saveobj() and equivalent to cv.Sim.save() or
    cv.Scenarios.save().

    **Examples**::

        cv.save('calib.sim', sim)
        cv.save(filename='school-closures.scens', folder='schools', obj=scens)
    '''
    filepath = sc.saveobj(*args, **kwargs)
    return filepath


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

    **Examples**::

        cv.git_info('covasim_version.json') # Writes to disk
        cv.git_info('covasim_version.json', check=True) # Checks that current version matches saved file
    '''
    info = sc.gitinfo(__file__)
    if not check: # Just get information
        if filename is not None:
            output = sc.savejson(filename, info, **kwargs)
        else:
            output = info
        return output
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
        return


def get_doubling_time(sim, series=None, interval=None, start_day=None, end_day=None, moving_window=None, exp_approx=False, max_doubling_time=100, eps=1e-3, verbose=None):
    '''
    Method to calculate doubling time.

    **Examples**::

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



def poisson_test(count1, count2, exposure1=1, exposure2=1, ratio_null=1,
                      method='score', alternative='two-sided'):
    '''Test for ratio of two sample Poisson intensities

    If the two Poisson rates are g1 and g2, then the Null hypothesis is

    H0: g1 / g2 = ratio_null

    against one of the following alternatives

    H1_2-sided: g1 / g2 != ratio_null
    H1_larger: g1 / g2 > ratio_null
    H1_smaller: g1 / g2 < ratio_null

    Args:
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

    Returns:
        pvalue two-sided # stat

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

    Author: Josef Perktold
    License: BSD-3

    destination statsmodels

    From: https://stackoverflow.com/questions/33944914/implementation-of-e-test-for-poisson-in-python

    Date: 2020feb24
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
            raise ValueError(f'invalid alternative "{alternative}"')
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
    else:
        raise ValueError(f'invalid method "{method}"')

    if dist == 'normal':
        return zstat_generic2(stat, 1, alternative)
    else:
        return pvalue#, stat
