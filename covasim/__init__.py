#%% Check imports -- note, must be manually updated to match requirements.txt unfortunately!

_min_sciris_version    = '0.16.0'
_min_scirisweb_version = '0.16.0'

# Check Sciris
try:
    import sciris as _sc
except ImportError:
    raise ImportError('Sciris not found; please install via "pip install sciris"')
if _sc.compareversions(_sc.__version__, _min_sciris_version) < 0:
    raise ImportError(f'Sciris {_sc.__version__} is incompatible; please upgrade via "pip install sciris=={_min_sciris_version}"')

# Check ScirisWeb
try:
    import scirisweb as _sw
except ImportError:
    raise ImportError('Scirisweb not found; please install via "pip install scirisweb"')
if _sc.compareversions(_sw.__version__, _min_scirisweb_version) < 0:
    raise ImportError(f'Scirisweb {_sw.__version__} is incompatible; please upgrade via "pip install scirisweb=={_min_scirisweb_version}"')

try:
    import covid_healthsystems as _hsys
except ImportError as E:
    print(f'Warning: covid_healthsystems is not available. Hospital capacity analyses will not be available. (Error: {str(E)})\n')
    _hsys = None

try:
    import parestlib as _pel
except ImportError as E:
    print(f'Warning: parestlib is not available. Automated calibration will not be available. (Error: {str(E)})\n')
    _pel = None



#%% Imports from here
from .cova_base.version import __version__, __versiondate__
from .cova_base import *
from .cova_generic import *