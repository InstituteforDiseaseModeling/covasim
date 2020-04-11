#%% Print version and license information
from .version import __version__, __versiondate__, __license__
print(__license__)

#%% Check that requirements are met
from . import requirements

#%% Import the actual model
from .utils         import *
from .defaults      import *
from .base          import *
from .parameters    import *
from .person        import *
from .population    import *
from .sim           import *
from .run           import *
from .interventions import *
