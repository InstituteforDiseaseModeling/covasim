#%% Print version and license information
from .version import __version__, __versiondate__, __license__
print(__license__)


#%% Check that requirements are met
from . import requirements

#%% Import the actual model
from .defaults      import * # No dependencies
from .parameters    import * # No dependencies
from .misc          import * # Depends on version
from .utils         import * # Depends on defaults
from .plotting      import * # Depends on defaults, misc
from .base          import * # Depends on version, misc, defaults, parameters, utils
from .people        import * # Depends on utils, defaults, base, plotting
from .population    import * # Depends on people et al.
from .interventions import * # Depends on defaults, utils, base
from .analysis      import * # Depends on utils, misc, interventions
from .sim           import * # Depends on almost everything
from .run           import * # Depends on sim
