#%% Print version and license information
from .version import __version__, __versiondate__, __license__
print(__license__)

#%% Check that requirements are met
from . import requirements

#%% Import the actual model
from .utils import *
from .base import *
from .parameters import *
from .people import *
from .sim import *
from .run import *
from .interventions import *
