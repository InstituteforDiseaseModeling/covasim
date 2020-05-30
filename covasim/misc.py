'''
Miscellaneous functions that do not belong anywhere else
'''

import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
import datetime as dt
import scipy.stats as sps
from . import version as cvver


__all__ = ['load_data', 'date', 'day', 'daydiff', 'load', 'save', 'savefig', 'get_png_metadata', 'git_info', 'check_version', 'check_save_info', 'get_doubling_time', 'poisson_test', 'compute_gof']


def load_data(datafile, columns=None, calculate=True, check_date=True, verbose=True, **kwargs):
    '''
    Load data for comparing to the model output, either from file or from a dataframe.

    Args:
        datafile (str or df): if a string, the name of the file to load (either Excel or CSV); if a dataframe, use directly
        columns (list): list of column names (otherwise, load all)
        calculate (bool): whether to calculate cumulative values from daily counts
        check_date (bool): whether to check that a 'date' column is present
        kwargs (dict): passed to pd.read_excel()

    Returns:
        data (dataframe): pandas dataframe of the loaded data
    '''

    # Load data
    if isinstance(datafile, str):
        if datafile.lower().endswith('csv'):
            raw_data = pd.read_csv(datafile, **kwargs)
        elif datafile.lower().endswith('xlsx'):
            raw_data = pd.read_excel(datafile, **kwargs)
        else:
            errormsg = f'Currently loading is only supported from .csv and .xlsx files, not {datafile}'
            raise NotImplementedError(errormsg)
    elif isinstance(datafile, pd.DataFrame):
        raw_data = datafile
    else:
        errormsg = f'Could not interpret data {type(datafile)}: must be a string or a dataframe'
        raise TypeError(errormsg)

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

    # Ensure required columns are present and reset the index
    if check_date:
        if 'date' not in data.columns:
            errormsg = f'Required column "date" not found; columns are {data.columns}'
            raise ValueError(errormsg)
        else:
            data['date'] = pd.to_datetime(data['date']).dt.date
        data.set_index('date', inplace=True, drop=False) # Don't drop so sim.data['date'] can still be accessed

    return data


def date(obj, *args, start_date=None, dateformat=None, as_date=True):
    '''
    Convert a string or a datetime object to a date object. To convert to an integer
    from the start day, you must supply a start date, or use sim.date() instead.

    Args:
        obj (str, date, datetime): the object to convert
        args (str, date, datetime): additional objects to convert
        start_date (str, date, datetime): the starting date, if an integer is supplied
        dateformat (str): the format to return the date in
        as_date (bool): whether to return as a datetime date instead of a string

    Returns:
        dates (date or list): either a single date object, or a list of them

    **Examples**::

        cv.date('2020-04-05') # Returns datetime.date(2020, 4, 5)
        cv.date('2020-04-14', start_date='2020-04-04', as_date=False) # Returns 10
    '''

    # Convert to list and handle other inputs
    obj = sc.promotetolist(obj) # Ensure it's iterable
    obj.extend(args)
    if dateformat is None:
        dateformat = '%Y-%m-%d'

    dates = []
    for d in obj:
        try:
            if type(d) == dt.date: # Do not use isinstance, since must be the exact type
                pass
            elif sc.isstring(d):
                d = sc.readdate(d).date()
            elif isinstance(d, dt.datetime):
                d = d.date()
            elif sc.isnumber(d):
                if start_date is None:
                    errormsg = f'To convert the number {d} to a date, you must supply start_date'
                    raise ValueError(errormsg)
                d = date(start_date) + dt.timedelta(days=int(d))
            else:
                errormsg = f'Cannot interpret {type(d)} as a date, must be date, datetime, or string'
                raise TypeError(errormsg)
            if as_date:
                dates.append(d)
            else:
                dates.append(d.strftime(dateformat))
        except Exception as E:
            errormsg = f'Conversion of "{d}" to a date failed: {str(E)}'
            raise ValueError(errormsg)

    # Return an integer rather than a list if only one provided
    if len(dates)==1:
        dates = dates[0]

    return dates


def day(obj, *args, start_day=None):
    '''
    Convert a string, date/datetime object, or int to a day (int), the number of
    days since the start day. See also date() and daydiff(). Used primarily via
    sim.day() rather than directly.

    Args:
        obj (str, date, int, or list): convert any of these objects to a day relative to the start day
        args (list): additional days
        start_day (str or date): the start day

    Returns:
        days (int or str): the day(s) in simulation time

    **Example**::

        sim.day('2020-04-05') # Returns 35
    '''

    # Do not process a day if it's not supplied
    if obj is None:
        return None

    # Convert to list
    if sc.isstring(obj) or sc.isnumber(obj) or isinstance(obj, (dt.date, dt.datetime)):
        obj = sc.promotetolist(obj) # Ensure it's iterable
    obj.extend(args)

    days = []
    for d in obj:
        if sc.isnumber(d):
            days.append(int(d)) # Just convert to an integer
        else:
            try:
                if sc.isstring(d):
                    d = sc.readdate(d).date()
                elif isinstance(d, dt.datetime):
                    d = d.date()
                d_day = (d - date(start_day)).days # Heavy lifting -- actually compute the day
                days.append(d_day)
            except Exception as E:
                errormsg = f'Could not interpret "{d}" as a date: {str(E)}'
                raise ValueError(errormsg)

    # Return an integer rather than a list if only one provided
    if len(days)==1:
        days = days[0]

    return days


def daydiff(*args):
    '''
    Convenience function to find the difference between two or more days. With
    only one argument, calculate days since 2020-01-01.

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


def get_caller(frame=2, tostring=True):
        '''
        Try to get information on the calling function, but fail gracefully.

        Frame 1 is the current file (this one), so not very useful. Frame 2 is
        the default assuming it is being called directly. Frame 3 is used if
        another function is calling this function internally.

        Args:
            frame (int): how many frames to descend (e.g. the caller of the caller of the...)
            tostring (bool): whether to return a string instead of a dict

        Returns:
            output (str/dict): the filename and line number of the calling function, either as a string or dict
        '''
        try:
            import inspect
            result = inspect.getouterframes(inspect.currentframe(), 2)
            fname = str(result[frame][1])
            lineno = str(result[frame][2])
            if tostring:
                output = f'{fname}, line {lineno}'
            else:
                output = {'filename':fname, 'lineno':lineno}
        except Exception as E:
            if tostring:
                output = f'Calling function information not available ({str(E)})'
            else:
                output = {'filename':'N/A', 'lineno':'N/A'}
        return output


def savefig(filename=None, dpi=None, comments=None, **kwargs):
    '''
    Wrapper for Matplotlib's savefig() function which automatically stores Covasim
    metadata in the figure. By default, saves

    Args:
        filename (str): name of the file to save to (default, timestamp)
        dpi (int): resolution of image (default 150)
        comments (str): additional metadata to save to the figure
        kwargs (dict): passed to savefig()

    **Example**::

        cv.Sim().run(do_plot=True)
        filename = cv.savefig()
    '''

    # Handle inputs
    dpi = kwargs.get('dpi', 150)
    metadata = kwargs.get('metadata', {})

    if filename is None:
        now = sc.getdate(dateformat='%Y-%b-%d_%H.%M.%S')
        filename = f'covasim_{now}.png'

    metadata = {}
    metadata['Covasim version'] = cvver.__version__
    gitinfo = git_info()
    for key,value in gitinfo.items():
        metadata[f'Covasim {key}'] = value
    metadata['Covasim current time'] = sc.getdate()
    metadata['Covasim calling file'] = get_caller()
    if comments:
        metadata['Covasim comments'] = comments

    # Save the figure
    pl.savefig(filename, dpi=dpi, metadata=metadata, **kwargs)
    return filename


def get_png_metadata(filename, output=False):
    '''
    Read metadata from a PNG file. For use with images saved with cv.savefig().
    Requires pillow, an optional dependency.

    Args:
        filename (str): the name of the file to load the data from

    **Example**::

        cv.Sim().run(do_plot=True)
        cv.savefig('covasim.png')
        cv.get_png_metadata('covasim.png')
    '''
    try:
        import PIL
    except ImportError as E:
        errormsg = f'Pillow import failed ({str(E)}), please install first (pip install pillow)'
        raise ImportError(errormsg)
    im = PIL.Image.open(filename)
    metadata = {}
    for key,value in im.info.items():
        if key.startswith('Covasim'):
            metadata[key] = value
            if not output:
                print(f'{key}: {value}')
    if output:
        return metadata
    else:
        return


def git_info(filename=None, check=False, comments=None, old_info=None, die=False, indent=2, verbose=True, **kwargs):
    '''
    Get current git information and optionally write it to disk. Simplest usage
    is cv.git_info(__file__)

    Args:
        filename (str): name of the file to write to or read from
        check (bool): whether or not to compare two git versions
        comments (str/dict): additional comments to include in the file
        old_info (dict): dictionary of information to check against
        die (bool): whether or not to raise an exception if the check fails
        indent (int): how many indents to use when writing the file to disk
        verbose (bool): detail to print
        kwargs (dict): passed to loadjson (if check=True) or loadjson (if check=False)

    **Examples**::

        cv.git_info() # Return information
        cv.git_info(__file__) # Writes to disk
        cv.git_info('covasim_version.gitinfo') # Writes to disk
        cv.git_info('covasim_version.gitinfo', check=True) # Checks that current version matches saved file
    '''

    # Handle the case where __file__ is supplied as the argument
    if isinstance(filename, str) and filename.endswith('.py'):
        filename = filename.replace('.py', '.gitinfo')

    # Get git info
    calling_file = sc.makefilepath(get_caller(frame=3, tostring=False)['filename'])
    cv_info = {'version':cvver.__version__}
    cv_info.update(sc.gitinfo(__file__, verbose=False))
    caller_info = sc.gitinfo(calling_file, verbose=False)
    caller_info['filename'] = calling_file
    info = {'covasim':cv_info, 'called_by':caller_info}
    if comments:
        info['comments'] = comments

    # Just get information and optionally write to disk
    if not check:
        if filename is not None:
            output = sc.savejson(filename, info, indent=indent, **kwargs)
        else:
            output = info
        return output

    # Check if versions match, and optionally raise an error
    else:
        if filename is not None:
            old_info = sc.loadjson(filename, **kwargs)
        string = ''
        old_cv_info = old_info['covasim'] if 'covasim' in old_info else old_info
        if cv_info != old_cv_info:
            string = f'Git information differs: {cv_info} vs. {old_cv_info}'
            if die:
                raise ValueError(string)
            elif verbose:
                print(string)
        return


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


def check_save_info(expected=None, filename=None, **kwargs):
    '''
    A convenience function that bundles check_version with git_info and saves
    automatically to disk from the calling file. The idea is to put this at the
    top of an analysis script, and commit the resulting file, to keep track of
    which version of Covasim was used.

    Args:
        expected (str): expected version information
        filename (str): file to save to; if None, guess based on current file name
        kwargs (dict): passed to git_info()

    **Examples**::

        cv.check_save_info()
        cv.check_save_info('1.3.2', filename='script.gitinfo', comments='This is the main analysis script')
    '''

    # First, check the version if supplied
    if expected:
        check_version(expected)

    # Now, check and save the git info
    if filename is None:
        filename = get_caller(tostring=False)['filename']
    git_info(filename=filename, **kwargs)

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


def compute_gof(actual, predicted, normalize=True, use_frac=False, use_squared=False, as_scalar='none', eps=1e-9, skestimator=None, **kwargs):
    '''
    Calculate the goodness of fit. By default use normalized absolute error, but
    highly customizable. For example, mean squared error is equivalent to
    setting normalize=False, use_squared=True, as_scalar='mean'.

    Args:
        actual      (arr):   array of actual (data) points
        predicted   (arr):   corresponding array of predicted (model) points
        normalize   (bool):  whether to divide the values by the largest value in either series
        use_frac    (bool):  convert to fractional mismatches rather than absolute
        use_squared (bool):  square the mismatches
        as_scalar   (str):   return as a scalar instead of a time series: choices are sum, mean, median
        eps         (float): to avoid divide-by-zero
        skestimator (str):   if provided, use this scikit-learn estimator instead
        kwargs      (dict):  passed to the scikit-learn estimator

    Returns:
        gofs (arr): array of goodness-of-fit values, or a single value if as_scalar is True

    **Examples**::

        x1 = np.cumsum(np.random.random(100))
        x2 = np.cumsum(np.random.random(100))

        e1 = compute_gof(x1, x2) # Default, normalized absolute error
        e2 = compute_gof(x1, x2, normalize=False, use_frac=False) # Fractional error
        e3 = compute_gof(x1, x2, normalize=False, use_squared=True, as_scalar='mean') # Mean squared error
        e4 = compute_gof(x1, x2, estimator='mean_squared_error') # Scikit-learn's MSE method
        e5 = compute_gof(x1, x2, as_scalar='median') # Normalized median absolute error -- highly robust
    '''

    # Custom estimator is supplied: use that
    if skestimator is not None:
        try:
            import sklearn.metrics as sm
            sklearn_gof = getattr(sm, skestimator) # Shortcut to e.g. sklearn.metrics.max_error
        except ImportError as E:
            raise ImportError(f'You must have scikit-learn >=0.22.2 installed: {str(E)}')
        except AttributeError:
            raise AttributeError(f'Estimator {skestimator} is not available; see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter for options')
        gof = sklearn_gof(actual, predicted, **kwargs)
        return gof

    # Default case: calculate it manually
    else:
        # Key step -- calculate the mismatch!
        gofs = abs(np.array(actual) - np.array(predicted))

        if normalize and not use_frac:
            maxmax = max(abs(actual).max(), abs(predicted).max())
            if maxmax>0:
                gofs /= maxmax

        if use_frac:
            if (actual<0).any() or (predicted<0).any():
                print('Warning: Calculating fractional errors for non-positive quantities is ill-advised!')
            else:
                maxvals = np.maximum(actual, predicted) + eps
                gofs /= maxvals

        if use_squared:
            gofs = gofs**2

        if as_scalar == 'sum':
            gofs = np.sum(gofs)
        elif as_scalar == 'mean':
            gofs = np.mean(gofs)
        elif as_scalar == 'median':
            gofs = np.median(gofs)

        return gofs