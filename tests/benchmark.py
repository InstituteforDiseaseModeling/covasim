# Benchmark the simulation

import sciris as sc
import covasim as cova

sim = cova.Sim()
to_profile = 'sim' # Must be one of the options listed below...currently only 1

func_options = {'sim':    sim.run,
                }

sc.profile(run=sim.run, follow=func_options[to_profile])
