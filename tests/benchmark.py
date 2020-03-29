# Benchmark the simulation

import sciris as sc
import covasim as cova

sim = cova.Sim()
sim['n_days'] = 60
to_profile = 'run' # Must be one of the options listed below...currently only 1

func_options = {'run':        sim.run,
                'initialize': sim.initialize,
                'init_people': sim.init_people,
                }

sc.profile(run=sim.run, follow=func_options[to_profile])
