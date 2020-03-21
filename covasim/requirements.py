import os

_min_sciris_version    = '0.16.0'

class Available:
    ''' Store information about optional imports '''
    scirisweb     = None
    healthsystems = None
    synthpops     = None
    parestlib     = None

available = Available() # Make this available at the module level
min_versions = {}

def get_requirements():
    filename = 'requirements.txt'
    cwd = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(os.pardir, cwd, filename)

    # Load requirements from txt file
    with open(filepath) as f:
        requirements = f.read().splitlines()





# Check Sciris
try:
    import sciris as _sc
except ImportError:
    raise ImportError('Sciris not found; please install via "pip install sciris"')
if _sc.compareversions(_sc.__version__, _min_sciris_version) < 0:
    raise ImportError(f'Sciris {_sc.__version__} is incompatible; please upgrade via "pip install sciris=={_min_sciris_version}"')


_min_scirisweb_version = '0.16.0'

# Check ScirisWeb import
try:
    import scirisweb as sw
except ImportError:
    raise ImportError('Scirisweb not found; please install via "pip install scirisweb"')
if _sc.compareversions(_sw.__version__, _min_scirisweb_version) < 0:
    raise ImportError(f'Scirisweb {_sw.__version__} is incompatible; please upgrade via "pip install scirisweb=={_min_scirisweb_version}"')


# Check health systems -- optional dependency
try:
    import covid_healthsystems as _hsys_available
    _hsys_available = True
except ImportError as E:
    print(f'Warning: covid_healthsystems is not available. Hospital capacity analyses will not be available. (Error: {str(E)})\n')
    _hsys_available = False


# Check synthpops -- optional dependency
try:
    import synthpops as _synth_available
    _synth_available = True
except ImportError as E:
    print(f'Warning: synthpops is not available. Detailed demographic data will not be available. (Error: {str(E)})\n')
    _synth_available = False

# Check parestlib -- optional dependency
try:
    import parestlib as _parest_available
    _parest_available = True
except ImportError as E:
    print(f'Warning: parestlib is not available. Automated calibration will not be available. (Error: {str(E)})\n')
    _parest_available = False


# Tidy up temporary variables, leaving _hsys and _parest since these are used later
del _min_sciris_version, _sc