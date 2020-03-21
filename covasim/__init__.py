# -*- coding: utf-8 -*-


#%% Print version and license information
from .version import __version__, __versiondate__, __license__
from . import requirements as _requirements
print(__license__)


#%% Check that requirements are met
_requirements.check_sciris()
_requirements.check_scirisweb(die=False)
_requirements.check_extra_libs()


#%% Imports from here -- just the framework, basic functions, and base -- the "base" version
from .framework import *
from .base import *
