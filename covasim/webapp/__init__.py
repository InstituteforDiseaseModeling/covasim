from covasim import __version__, __versiondate__ # These are ignored by star imports

_min_scirisweb_version = '0.16.0'

# Check ScirisWeb import
try:
    import scirisweb as _sw
except ImportError:
    raise ImportError('Scirisweb not found; please install via "pip install scirisweb"')
if _sc.compareversions(_sw.__version__, _min_scirisweb_version) < 0:
    raise ImportError(f'Scirisweb {_sw.__version__} is incompatible; please upgrade via "pip install scirisweb=={_min_scirisweb_version}"')


from covasim.framework import *
from .parameters import *
from .model import *