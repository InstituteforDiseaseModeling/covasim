#%% Print version and license information
from .version import __version__, __versiondate__, __license__
print(__license__)

#%% Check that requirements are met
from . import requirements
requirements.check_sciris()
requirements.check_scirisweb(die=False)
requirements.check_extra_libs()

#%% Import the actual model
from .utils import *
from .base import *
from .parameters import *
from .model import *
from .run import *