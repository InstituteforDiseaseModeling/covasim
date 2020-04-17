Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a
Changelog <https://keepachangelog.com/en/1.0.0/>`__, and this project
adheres to `Semantic
Versioning <https://semver.org/spec/v2.0.0.html>`__.


Version 0.27.5 (2020-04-17)
---------------------------
- Caches Numba functions, reducing load time from 2.5 to 0.5 seconds
- Fixed issue with saving populations in scenarios
- Refactored how populations are handled, removing ``use_layers`` parameter (use ``pop_type`` instead)


Version 0.27.1 (2020-04-17)
---------------------------
- Added documentation


Version 0.27.0 (2020-04-16)
---------------------------
-  Refactor calculations to be vector-based rather than object based
-  Include factors for per-person viral load (transmissibility) and
   susceptibility
-  Started a changelog (needless to say)