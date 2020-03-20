# -*- coding: utf-8 -*-


#%% Print version and license information
from .version import __version__, __versiondate__, __license__
print(__license__)


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


# Check health systems -- optional dependency
try:
    import covid_healthsystems as _hsys
except ImportError as E:
    print(f'Warning: covid_healthsystems is not available. Hospital capacity analyses will not be available. (Error: {str(E)})\n')
    _hsys = None


# Check parestlib -- optional dependency
try:
    import parestlib as _parest
except ImportError as E:
    print(f'Warning: parestlib is not available. Automated calibration will not be available. (Error: {str(E)})\n')
    _parest = None


# Tidy up temporary variables, leaving _hsys and _parest since these are used later
del _min_sciris_version, _min_scirisweb_version, _sc, _sw


#%% Imports from here
from .cova_base import *
from .cova_generic import *