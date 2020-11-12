# Check that requirements are met and set options
from . import requirements
from . import options

# Import the version and print the license unless verbosity is disabled
from .version import __version__, __versiondate__, __license__
if options.verbose:
    print(__license__)

# Import the actual model
from .defaults      import * # Depends on options
from .parameters    import * # Depends on options
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
