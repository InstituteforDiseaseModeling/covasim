print('hi')

import sciris as sc

t = sc.tic()

sc.toc(t, label='A0')

from covid_abm.version import *
sc.toc(t, label='Aa')
from covid_abm.utils import *
sc.toc(t, label='Ab')
from covid_abm.poisson_stats import *
sc.toc(t, label='Ac')
from .parameters import *
sc.toc(t, label='Ad')
from .model import *
sc.toc(t, label='Ae')

