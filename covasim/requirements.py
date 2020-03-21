#%% Housekeeping

import os

__all__ = ['available', 'min_versions', 'get_min_versions', 'check_sciris', 'check_scirisweb', 'check_extra_libs']


available = {} # Make this available at the module level
min_versions = {}

#%% Get the right versions

def get_min_versions():
    ''' Parse requirements.txt and get minimum versions for everything '''

    # Set options and get the file to load
    filename = 'requirements.txt'
    comparator = '>='
    cwd = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(cwd, os.pardir, filename)

    # Load requirements from file
    with open(filepath) as f:
        requirements = f.read().splitlines()

    # Process them
    for requirement in requirements: # Expecting e.g. 'scirisweb>=0.16.0'
        split = requirement.split(comparator)
        if len(split) == 2:
            package = split[0]
            version = split[1]
            min_versions[package] = version

    return # NB, modifies the module-level min_versions dict in-place

get_min_versions() # Populate these straight away


#%% Check dependencies

def check_sciris():
    ''' Check that Sciris is available and the right version '''
    try:
        import sciris as sc
    except ModuleNotFoundError:
        raise ModuleNotFoundError('Sciris is a required dependency but is not found; please install via "pip install sciris"')
    ver = sc.__version__
    minver = min_versions['sciris']
    if sc.compareversions(ver, minver) < 0:
        raise ImportError(f'You have Sciris {ver} but {minver} is required; please upgrade via "pip install --upgrade sciris"')
    return


def check_scirisweb(die=False):
    ''' Check that Scirisweb is available and the right version '''
    import sciris as sc # Here since one of the purposes of this file is to check if this exists
    available['scirisweb'] = True # Assume it works
    import_error = ''
    version_error = ''

    # Try imports
    try:
        import scirisweb
    except ModuleNotFoundError:
        import_error = 'Scirisweb not found; please install via "pip install scirisweb"'
    if not import_error:
        ver = scirisweb.__version__
        minver = min_versions['scirisweb']
        if sc.compareversions(ver, minver) < 0:
            version_error = f'You have Scirisweb {ver} but {minver} is required; please upgrade via "pip install --upgrade scirisweb"'

    # Handle consequences
    if die:
        if import_error:
            raise ModuleNotFoundError(import_error)
        elif version_error:
            raise ImportError(version_error)
    else:
        if import_error:
            print('Warning: scirisweb was not found; webapp functionality is not available (you can install with "pip install scirisweb")')
        elif version_error:
            print(f'Warning: scirisweb is version {ver} but {minver} is required; webapp is disabled (fix with "pip install --upgrade scirisweb")')

    if import_error or version_error:
        available['scirisweb'] = False

    return


def check_extra_libs():
    ''' Check whether optional dependencies are available '''

    # Check health systems -- optional dependency
    try:
        import covid_healthsystems # noqa
        available['covid_healthsystems'] = True
    except ImportError as E:
        import_error = f'Warning: covid_healthsystems is not available. Hospital capacity analyses will not be available. (Error: {str(E)})\n'
        available['covid_healthsystems'] = False
        print(import_error)

    # Check synthpops -- optional dependency
    try:
        import synthpops # noqa
        available['synthpops'] = True
    except ImportError as E:
        import_error = f'Warning: synthpops is not available. Detailed demographic data will not be available. (Error: {str(E)})\n'
        available['synthpops'] = True
        print(import_error)

    # Check parestlib -- optional dependency
    try:
        import parestlib as _parest_available # noqa
        available['parestlib'] = True
    except ImportError as E:
        import_error = f'Warning: parestlib is not available. Automated calibration will not be available. (Error: {str(E)})\n'
        available['parestlib'] = True
        print(import_error)

    return

