'''
Check that correct versions of each library are installed, and print warnings
or errors if not.
'''

#%% Housekeeping

__all__ = ['available', 'min_versions', 'check_sciris', 'check_scirisweb', 'check_extra_libs']


available = {} # Make this available at the module level
min_versions = {'sciris':'0.16.7', 'scirisweb':'0.16.0'}


#%% Check dependencies

def check_sciris():
    ''' Check that Sciris is available and the right version '''
    try:
        import sciris as sc
    except ModuleNotFoundError:
        errormsg = 'Sciris is a required dependency but is not found; please install via "pip install sciris"'
        raise ModuleNotFoundError(errormsg)
    ver = sc.__version__
    minver = min_versions['sciris']
    if sc.compareversions(ver, minver) < 0:
        errormsg = f'You have Sciris {ver} but {minver} is required; please upgrade via "pip install --upgrade sciris"'
        raise ImportError(errormsg)
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
        import_error = 'Scirisweb not found; please rerun setup.py or install via "pip install scirisweb"'
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

    # Check synthpops -- optional dependency
    try:
        import synthpops # noqa
        available['synthpops'] = True
    except ImportError as E:
        import_error = f'Note: synthpops (for detailed demographic data) is not available ({str(E)})\n'
        available['synthpops'] = True
        print(import_error)

    # # Check parestlib -- optional dependency -- not currently implemented
    # try:
    #     import parestlib as _parest_available # noqa
    #     available['parestlib'] = True
    # except ImportError as E:
    #     import_error = f'Note: parestlib (for automatic calibration) is not available ({str(E)})\n'
    #     available['parestlib'] = True
    #     print(import_error)

    return

# Perform the version checks on import
check_sciris()
check_scirisweb(die=False)
check_extra_libs()