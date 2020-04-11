#%% Print version and license information
from .version import __version__, __versiondate__, __license__
print(__license__)

import sciris as sc

sc.tic()
print('a')

#%% Check that requirements are met
from . import requirements
sc.toc()
print('b')

#%% Import the actual model
from .utils         import *
sc.toc()
print('c')
from .defaults      import *
sc.toc()
from .base          import *
sc.toc()
from .parameters    import *
from .person        import *
from .population    import *
from .sim           import *
from .run           import *
from .interventions import *
