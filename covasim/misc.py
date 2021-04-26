'''
Miscellaneous functions that do not belong anywhere else
'''

import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
import scipy.stats as sps
from pathlib import Path
from . import version as cvv
from distutils.version import LooseVersion

#%% Convenience imports from Sciris

__all__ = ['date', 'day', 'daydiff', 'date_range']

date       = sc.date
day        = sc.day
daydiff    = sc.daydiff
date_range = sc.daterange


#%% Loading/saving functions

__all__ += ['load_data', 'load', 'save', 'savefig']


def load_data(datafile, columns=None, calculate=True, check_date=True, verbose=True, start_day=None, **kwargs):
    '''
    Load data for comparing to the model output, either from file or from a dataframe.

    Args:
        datafile (str or df): if a string, the name of the file to load (either Excel or CSV); if a dataframe, use directly
        columns (list): list of column names (otherwise, load all)
        calculate (bool): whether to calculate cumulative values from daily counts
        check_date (bool): whether to check that a 'date' column is present
        start_day (date): if the 'date' column is provided as integer number of days, consider them relative to this
        kwargs (dict): passed to pd.read_excel()

    Returns:
        data (dataframe): pandas dataframe of the loaded data
    '''

    # Load data
    if isinstance(datafile, Path): # Convert to a string
        datafile = str(datafile)
    if isinstance(datafile, str):
        df_lower = datafile.lower()
        if df_lower.endswith('csv'):
            raw_data = pd.read_csv(datafile, **kwargs)
        elif df_lower.endswith('xlsx') or df_lower.endswith('xls'):
            raw_data = pd.read_excel(datafile, **kwargs)
        elif df_lower.endswith('json'):
            raw_data = pd.read_json(datafile, **kwargs)
        else:
            errormsg = f'Currently loading is only supported from .csv, .xls/.xlsx, and .json files, not "{datafile}"'
            raise NotImplementedError(errormsg)
    elif isinstance(datafile, pd.DataFrame):
        raw_data = datafile
    else: # pragma: no cover
        errormsg = f'Could not interpret data {type(datafile)}: must be a string or a dataframe'
        raise TypeError(errormsg)

    # Confirm data integrity and simplify
    if columns is not None:
        for col in columns:
            if col not in raw_data.columns: # pragma: no cover
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
            if data['date'].dtype == np.int64: # If it's integers, treat it as days from the start day
                data['date'] = sc.date(data['date'].values, start_date=start_day)
            else: # Otherwise, use Pandas to convert it
                data['date'] = pd.to_datetime(data['date']).dt.date
        data.set_index('date', inplace=True, drop=False) # Don't drop so sim.data['date'] can still be accessed

    return data


def load(*args, do_migrate=True, update=True, verbose=True, **kwargs):
    '''
    Convenience method for sc.loadobj() and equivalent to cv.Sim.load() or
    cv.Scenarios.load().

    Args:
        filename (str): file to load
        do_migrate (bool): whether to migrate if loading an old object
        update (bool): whether to modify the object to reflect the new version
        verbose (bool): whether to print migration information
        args (list): passed to sc.loadobj()
        kwargs (dict): passed to sc.loadobj()

    Returns:
        Loaded object

    **Examples**::

        sim = cv.load('calib.sim') # Equivalent to cv.Sim.load('calib.sim')
        scens = cv.load(filename='school-closures.scens', folder='schools')
    '''
    obj = sc.loadobj(*args, **kwargs)
    if hasattr(obj, 'version'):
        v_curr = cvv.__version__
        v_obj = obj.version
        cmp = check_version(v_obj, verbose=False)
        if cmp != 0:
            print(f'Note: you have Covasim v{v_curr}, but are loading an object from v{v_obj}')
            if do_migrate:
                obj = migrate(obj, update=update, verbose=verbose)
    return obj


def save(*args, **kwargs):
    '''
    Convenience method for sc.saveobj() and equivalent to cv.Sim.save() or
    cv.Scenarios.save().

    Args:
        filename (str): file to save to
        obj (object): object to save
        args (list): passed to sc.saveobj()
        kwargs (dict): passed to sc.saveobj()

    Returns:
        Filename the object is saved to

    **Examples**::

        cv.save('calib.sim', sim) # Equivalent to sim.save('calib.sim')
        cv.save(filename='school-closures.scens', folder='schools', obj=scens)
    '''
    filepath = sc.saveobj(*args, **kwargs)
    return filepath


def savefig(filename=None, comments=None, **kwargs):
    '''
    Wrapper for Matplotlib's savefig() function which automatically stores Covasim
    metadata in the figure. By default, saves (git) information from both the Covasim
    version and the calling function. Additional comments can be added to the saved
    file as well. These can be retrieved via cv.get_png_metadata(). Metadata can
    also be stored for SVG and PDF formats, but cannot be automatically retrieved.

    Args:
        filename (str): name of the file to save to (default, timestamp)
        comments (str): additional metadata to save to the figure
        kwargs (dict): passed to savefig()

    **Example**::

        cv.Sim().run(do_plot=True)
        filename = cv.savefig()
    '''

    # Handle inputs
    dpi = kwargs.pop('dpi', 150)
    metadata = kwargs.pop('metadata', {})

    if filename is None: # pragma: no cover
        now = sc.getdate(dateformat='%Y-%b-%d_%H.%M.%S')
        filename = f'covasim_{now}.png'

    metadata = {}
    metadata['Covasim version'] = cvv.__version__
    gitinfo = git_info()
    for key,value in gitinfo['covasim'].items():
        metadata[f'Covasim {key}'] = value
    for key,value in gitinfo['called_by'].items():
        metadata[f'Covasim caller {key}'] = value
    metadata['Covasim current time'] = sc.getdate()
    metadata['Covasim calling file'] = sc.getcaller()
    if comments:
        metadata['Covasim comments'] = comments

    # Handle different formats
    lcfn = filename.lower() # Lowercase filename
    if lcfn.endswith('pdf') or lcfn.endswith('svg'):
        metadata = {'Keywords':str(metadata)} # PDF and SVG doesn't support storing a dict

    # Save the figure
    pl.savefig(filename, dpi=dpi, metadata=metadata, **kwargs)
    return filename


#%% Migration functions

__all__ += ['migrate']

def migrate_lognormal(pars, revert=False, verbose=True):
    '''
    Small helper function to automatically migrate the standard deviation of lognormal
    distributions to match pre-v2.1.0 runs (where it was treated as the variance instead).
    To undo the migration, run with revert=True.

    Args:
        pars (dict): the parameters dictionary; or, alternatively, the sim object the parameters will be taken from
        revert (bool): whether to reverse the update rather than make it
        verbose (bool): whether to print out the old and new values
    '''
    # Handle different input types
    from . import base as cvb
    if isinstance(pars, cvb.BaseSim):
        pars = pars.pars # It's actually a sim, not a pars object

    # Convert each value to the square root, since squared in the new version
    for key,dur in pars['dur'].items():
        if 'lognormal' in dur['dist']:
            old = dur['par2']
            if revert:
                new = old**2
            else:
                new = np.sqrt(old)
            dur['par2'] = new
            if verbose > 1:
                print(f'  Updating {key} std from {old:0.2f} to {new:0.2f}')

    # Store whether migration has occurred so we don't accidentally do it twice
    if not revert:
        pars['migrated_lognormal'] = True
    else:
        pars.pop('migrated_lognormal', None)

    return


def migrate_strains(pars, verbose=True):
    '''
    Small helper function to add necessary strain parameters.
    '''
    pars['use_waning'] = False
    pars['n_strains'] = 1
    pars['n_strains'] = 1
    pars['strains'] = []
    return


def migrate(obj, update=True, verbose=True, die=False):
    '''
    Define migrations allowing compatibility between different versions of saved
    files. Usually invoked automatically upon load, but can be called directly by
    the user to load custom objects, e.g. lists of sims.

    Currently supported objects are sims, multisims, scenarios, and people.

    Args:
        obj (any): the object to migrate
        update (bool): whether to update version information to current version after successful migration
        verbose (bool): whether to print warnings if something goes wrong
        die (bool): whether to raise an exception if something goes wrong

    Returns:
        The migrated object

    **Example**::

        sims = cv.load('my-list-of-sims.obj')
        sims = [cv.migrate(sim) for sim in sims]
    '''
    # Import here to avoid recursion
    from . import base as cvb
    from . import run as cvr
    from . import interventions as cvi

    # Migrations for simulations
    if isinstance(obj, cvb.BaseSim):
        sim = obj

        # Migration from <2.0.0 to 2.0.0
        if sc.compareversions(sim.version, '2.0.0') == -1: # Migrate from <2.0 to 2.0
            if verbose: print(f'Migrating sim from version {sim.version} to version {cvv.__version__}')

            # Add missing attribute
            if not hasattr(sim, '_default_ver'):
                sim._default_ver = None

            # Recursively migrate people if needed
            if sim.people:
                sim.people = migrate(sim.people, update=update)

            # Rename intervention attribute
            tps = sim.get_interventions(cvi.test_prob)
            for tp in tps: # pragma: no cover
                try:
                    tp.sensitivity = tp.test_sensitivity
                    del tp.test_sensitivity
                except:
                    pass

        # Migration from <2.1.0 to 2.1.0
        if sc.compareversions(sim.version, '2.1.0') == -1:
            if verbose:
                print(f'Migrating sim from version {sim.version} to version {cvv.__version__}')
                print('Note: updating lognormal stds to restore previous behavior; see v2.1.0 changelog for details')
            migrate_lognormal(sim.pars, verbose=verbose)

        # Migration from <3.0.0 to 3.0.0
        if sc.compareversions(sim.version, '3.0.0') == -1:
            if verbose:
                print(f'Migrating sim from version {sim.version} to version {cvv.__version__}')
                print('Adding strain parameters')
            migrate_strains(sim.pars, verbose=verbose)

    # Migrations for People
    elif isinstance(obj, cvb.BasePeople): # pragma: no cover
        ppl = obj
        if not hasattr(ppl, 'version'): # For people prior to 2.0
            if verbose: print(f'Migrating people from version <2.0 to version {cvv.__version__}')
            cvb.set_metadata(ppl) # Set all metadata

    # Migrations for MultiSims -- use recursion
    elif isinstance(obj, cvr.MultiSim):
        msim = obj
        msim.base_sim = migrate(msim.base_sim, update=update)
        msim.sims = [migrate(sim, update=update) for sim in msim.sims]
        if not hasattr(msim, 'version'): # For msims prior to 2.0
            if verbose: print(f'Migrating multisim from version <2.0 to version {cvv.__version__}')
            cvb.set_metadata(msim) # Set all metadata
            msim.label = None

    # Migrations for Scenarios
    elif isinstance(obj, cvr.Scenarios):
        scens = obj
        scens.base_sim = migrate(scens.base_sim, update=update)
        for key,simlist in scens.sims.items():
            scens.sims[key] = [migrate(sim, update=update) for sim in simlist] # Nested loop
        if not hasattr(scens, 'version'): # For scenarios prior to 2.0
            if verbose: print(f'Migrating scenarios from version <2.0 to version {cvv.__version__}')
            cvb.set_metadata(scens) # Set all metadata
            scens.label = None

    # Unreconized object type
    else:
        errormsg = f'Object {obj} of type {type(obj)} is not understood and cannot be migrated: must be a sim, multisim, scenario, or people object'
        if die:
            raise TypeError(errormsg)
        elif verbose: # pragma: no cover
            print(errormsg)
            return


    # If requested, update the stored version to the current version
    if update:
        obj.version = cvv.__version__

    return obj



#%% Versioning functions

__all__ += ['git_info', 'check_version', 'check_save_version', 'get_version_pars', 'get_png_metadata']


def git_info(filename=None, check=False, comments=None, old_info=None, die=False, indent=2, verbose=True, frame=2, **kwargs):
    '''
    Get current git information and optionally write it to disk. Simplest usage
    is cv.git_info(__file__)

    Args:
        filename  (str): name of the file to write to or read from
        check    (bool): whether or not to compare two git versions
        comments (dict): additional comments to include in the file
        old_info (dict): dictionary of information to check against
        die      (bool): whether or not to raise an exception if the check fails
        indent    (int): how many indents to use when writing the file to disk
        verbose  (bool): detail to print
        frame     (int): how many frames back to look for caller info
        kwargs   (dict): passed to sc.loadjson() (if check=True) or sc.savejson() (if check=False)

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
    calling_file = sc.makefilepath(sc.getcaller(frame=frame, tostring=False)['filename'])
    cv_info = {'version':cvv.__version__}
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
        if cv_info != old_cv_info: # pragma: no cover
            string = f'Git information differs: {cv_info} vs. {old_cv_info}'
            if die:
                raise ValueError(string)
            elif verbose:
                print(string)
        return


def check_version(expected, die=False, verbose=True):
    '''
    Get current git information and optionally write it to disk. The expected
    version string may optionally start with '>=' or '<=' (== is implied otherwise),
    but other operators (e.g. ~=) are not supported. Note that e.g. '>' is interpreted
    to mean '>='.

    Args:
        expected (str): expected version information
        die (bool): whether or not to raise an exception if the check fails

    **Example**::

        cv.check_version('>=1.7.0', die=True) # Will raise an exception if an older version is used
    '''
    if expected.startswith('>'):
        valid = 1
    elif expected.startswith('<'):
        valid = -1
    else:
        valid = 0 # Assume == is the only valid comparison
    expected = expected.lstrip('<=>') # Remove comparator information
    version = cvv.__version__
    compare = sc.compareversions(version, expected) # Returns -1, 0, or 1
    relation = ['older', '', 'newer'][compare+1] # Picks the right string
    if relation: # Versions mismatch, print warning or raise error
        string = f'Note: Covasim is {relation} than expected ({version} vs. {expected})'
        if die and compare != valid:
            raise ValueError(string)
        elif verbose:
            print(string)
    return compare


def check_save_version(expected=None, filename=None, die=False, verbose=True, **kwargs):
    '''
    A convenience function that bundles check_version with git_info and saves
    automatically to disk from the calling file. The idea is to put this at the
    top of an analysis script, and commit the resulting file, to keep track of
    which version of Covasim was used.

    Args:
        expected (str): expected version information
        filename (str): file to save to; if None, guess based on current file name
        kwargs (dict): passed to git_info(), and thence to sc.savejson()

    **Examples**::

        cv.check_save_version()
        cv.check_save_version('1.3.2', filename='script.gitinfo', comments='This is the main analysis script')
        cv.check_save_version('1.7.2', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})
    '''

    # First, check the version if supplied
    if expected:
        check_version(expected, die=die, verbose=verbose)

    # Now, check and save the git info
    if filename is None:
        filename = sc.getcaller(tostring=False)['filename']
    git_info(filename=filename, frame=3, **kwargs)

    return


def get_version_pars(version, verbose=True):
    '''
    Function for loading parameters from the specified version.

    Parameters will be loaded for Covasim 'as at' the requested version i.e. the
    most recent set of parameters that is <= the requested version. Available
    parameter values are stored in the regression folder. If parameters are available
    for versions 1.3, and 1.4, then this function will return the following

    - If parameters for version '1.3' are requested, parameters will be returned from '1.3'
    - If parameters for version '1.3.5' are requested, parameters will be returned from '1.3', since
      Covasim at version 1.3.5 would have been using the parameters defined at version 1.3.
    - If parameters for version '1.4' are requested, parameters will be returned from '1.4'

    Args:
        version (str): the version to load parameters from

    Returns:
        Dictionary of parameters from that version
    '''

    # Construct a sorted list of available parameters based on the files in the regression folder
    regression_folder = sc.thisdir(__file__, 'regression', aspath=True)
    available_versions = [x.stem.replace('pars_v','') for x in regression_folder.iterdir() if x.suffix=='.json']
    available_versions = sorted(available_versions, key=LooseVersion)

    # Find the highest parameter version that is <= the requested version
    version_comparison = [sc.compareversions(version, v)>=0 for v in available_versions]
    try:
        target_version = available_versions[sc.findlast(version_comparison)]
    except IndexError:
        errormsg = f"Could not find a parameter version that was less than or equal to '{version}'. Available versions are {available_versions}"
        raise ValueError(errormsg)

    # Load the parameters
    pars = sc.loadjson(filename=regression_folder/f'pars_v{target_version}.json', folder=regression_folder)
    if verbose:
        print(f'Loaded parameters from {target_version}')

    return pars


def get_png_metadata(filename, output=False):
    '''
    Read metadata from a PNG file. For use with images saved with cv.savefig().
    Requires pillow, an optional dependency. Metadata retrieval for PDF and SVG
    is not currently supported.

    Args:
        filename (str): the name of the file to load the data from

    **Example**::

        cv.Sim().run(do_plot=True)
        cv.savefig('covasim.png')
        cv.get_png_metadata('covasim.png')
    '''
    try:
        import PIL
    except ImportError as E: # pragma: no cover
        errormsg = f'Pillow import failed ({str(E)}), please install first (pip install pillow)'
        raise ImportError(errormsg) from E
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



#%% Simulation/statistics functions

__all__ += ['get_doubling_time', 'poisson_test', 'compute_gof']


def get_doubling_time(sim, series=None, interval=None, start_day=None, end_day=None, moving_window=None, exp_approx=False, max_doubling_time=100, eps=1e-3, verbose=None):
    '''
    Alternate method to calculate doubling time (one is already implemented in
    the sim object).

    **Examples**::

        cv.get_doubling_time(sim, interval=[3,30]) # returns the doubling time over the given interval (single float)
        cv.get_doubling_time(sim, interval=[3,30], moving_window=3) # returns doubling times calculated over moving windows (array)
    '''

    # Set verbose level
    if verbose is None:
        verbose = sim['verbose']

    # Validate inputs: series
    if series is None or isinstance(series, str):
        if not sim.results_ready: # pragma: no cover
            raise Exception("Results not ready, cannot calculate doubling time")
        else:
            if series is None or series not in sim.result_keys():
                sc.printv("Series not supplied or not found in results; defaulting to use cumulative exposures", 1, verbose)
                series='cum_infections'
            series = sim.results[series].values
    else:
        series = sc.promotetoarray(series)

    # Validate inputs: interval
    if interval is not None:
        if len(interval) != 2: # pragma: no cover
            sc.printv(f"Interval should be a list/array/tuple of length 2, not {len(interval)}. Resetting to length of series.", 1, verbose)
            interval = [0,len(series)]
        start_day, end_day = interval[0], interval[1]

    if len(series) < end_day:
        sc.printv(f"End day {end_day} is after the series ends ({len(series)}). Resetting to length of series.", 1, verbose)
        end_day = len(series)
    int_length = end_day - start_day

    # Deal with moving window
    if moving_window is not None:
        if not sc.isnumber(moving_window): # pragma: no cover
            sc.printv("Moving window should be an integer; ignoring and calculating single result", 1, verbose)
            doubling_time = get_doubling_time(sim, series=series, start_day=start_day, end_day=end_day, moving_window=None, exp_approx=exp_approx)

        else:
            if not isinstance(moving_window,int): # pragma: no cover
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
            except ModuleNotFoundError as E: # pragma: no cover
                errormsg = f'Could not import statsmodels ({E}), falling back to exponential approximation'
                print(errormsg)
                exp_approx = True
        if exp_approx:
            if series[start_day] > 0:
                r = series[end_day] / series[start_day]
                if r > 1:
                    doubling_time = int_length * np.log(2) / np.log(r)
                    doubling_time = min(doubling_time, max_doubling_time)  # Otherwise, it's unbounded
            else: # pragma: no cover
                raise ValueError("Can't calculate doubling time with exponential approximation when initial value is zero.")
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
                else: # pragma: no cover
                    raise ValueError(f"Can't calculate doubling time for series {series[start_day:end_day]}. Check whether series is growing.")
            else: # pragma: no cover
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
        else: # pragma: no cover
            raise ValueError(f'Invalid alternative "{alternative}"')
        return pvalue

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


def compute_gof(actual, predicted, normalize=True, use_frac=False, use_squared=False, as_scalar='none', eps=1e-9, skestimator=None, estimator=None, **kwargs):
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
        estimator   (func):  if provided, use this custom estimator instead
        kwargs      (dict):  passed to the scikit-learn or custom estimator

    Returns:
        gofs (arr): array of goodness-of-fit values, or a single value if as_scalar is True

    **Examples**::

        x1 = np.cumsum(np.random.random(100))
        x2 = np.cumsum(np.random.random(100))

        e1 = compute_gof(x1, x2) # Default, normalized absolute error
        e2 = compute_gof(x1, x2, normalize=False, use_frac=False) # Fractional error
        e3 = compute_gof(x1, x2, normalize=False, use_squared=True, as_scalar='mean') # Mean squared error
        e4 = compute_gof(x1, x2, skestimator='mean_squared_error') # Scikit-learn's MSE method
        e5 = compute_gof(x1, x2, as_scalar='median') # Normalized median absolute error -- highly robust
    '''

    # Handle inputs
    actual    = np.array(sc.dcp(actual), dtype=float)
    predicted = np.array(sc.dcp(predicted), dtype=float)

    # Scikit-learn estimator is supplied: use that
    if skestimator is not None: # pragma: no cover
        try:
            import sklearn.metrics as sm
            sklearn_gof = getattr(sm, skestimator) # Shortcut to e.g. sklearn.metrics.max_error
        except ImportError as E:
            raise ImportError(f'You must have scikit-learn >=0.22.2 installed: {str(E)}')
        except AttributeError:
            raise AttributeError(f'Estimator {skestimator} is not available; see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter for options')
        gof = sklearn_gof(actual, predicted, **kwargs)
        return gof

    # Custom estimator is supplied: use that
    if estimator is not None:
        try:
            gof = estimator(actual, predicted, **kwargs)
        except Exception as E:
            errormsg = f'Custom estimator "{estimator}" must be a callable function that accepts actual and predicted arrays, plus optional kwargs'
            raise RuntimeError(errormsg) from E
        return gof

    # Default case: calculate it manually
    else:
        # Key step -- calculate the mismatch!
        gofs = abs(np.array(actual) - np.array(predicted))

        if normalize and not use_frac:
            actual_max = abs(actual).max()
            if actual_max>0:
                gofs /= actual_max

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