What's new
==========

All major changes to the codebase are documented in this file. Note: in many cases,
changes from multiple patch versions are grouped together, so numbering will not be
strictly consecutive. 


Version 0.27.9 (2020-04-17)
---------------------------
- Caches Numba functions, reducing load time from 2.5 to 0.5 seconds
- Pins Numba to 0.48, which is 10x faster than 0.49
- Fixed issue with saving populations in scenarios
- Refactored how populations are handled, removing ``use_layers`` parameter (use ``pop_type`` instead)
- Removed layer key from layer object, reducing total sim memory footprint by 3x
- Improved handling of mismatches between loaded population layers and simulation parameters
- Added custom key errors to handle multiline error messages
- Added `documentation <https://institutefordiseasemodeling.github.io/covasim-docs/>`__


Version 0.27.0 (2020-04-16)
---------------------------
-  Refactor calculations to be vector-based rather than object based
-  Include factors for per-person viral load (transmissibility) and
   susceptibility
-  Started a changelog (needless to say)