What's new
==========

All major changes to the codebase are documented in this file. Note: in many cases,
changes from multiple patch versions are grouped together, so numbering will not be
strictly consecutive.


Version 0.28.1 (2020-04-19)
----------------------------
- Major refactor of transmission trees, including additional detail via ``sim.people.transtree.make_detailed()``
- Counting of diagnoses before and after interventions on each timestep (allowing people to go into quarantine on the same day)
- Improved saving of people in scenarios, and updated keyword for sims (``sim.save(keep_people=True)``)


Version 0.28.0 (2020-04-19)
----------------------------
- Includes dynamic per-person viral load
- Refactored data types
- Changed how populations are handled, including adding a ``dynam_layer`` parameter to specify which layers are dynamic
- Disease progression duration parameters were updated to be longer
- Fixed bugs with quarantine
- Fixed bug with hybrid school and work contacts
- Changed contact tracing to be only for contacts with nonzero transmission


Version 0.27.12 (2020-04-17)
----------------------------
- Caches Numba functions, reducing load time from 2.5 to 0.5 seconds
- Pins Numba to 0.48, which is 10x faster than 0.49
- Fixed issue with saving populations in scenarios
- Refactored how populations are handled, removing ``use_layers`` parameter (use ``pop_type`` instead)
- Removed layer key from layer object, reducing total sim memory footprint by 3x
- Improved handling of mismatches between loaded population layers and simulation parameters
- Added custom key errors to handle multiline error messages
- Fix several issues with probability-based testing
- Changed how layer betas are applied (inside the sim rather than statically)
- Added more detail to the transmission tree
- Refactored random population calculation, speeding up large populations (>100k) by a factor of 10
- Added `documentation <https://institutefordiseasemodeling.github.io/covasim-docs/>`__


Version 0.27.0 (2020-04-16)
---------------------------
-  Refactor calculations to be vector-based rather than object based
-  Include factors for per-person viral load (transmissibility) and
   susceptibility
-  Started a changelog (needless to say)