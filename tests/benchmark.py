# Benchmark the simulation

import sciris as sc
import covid_seattle

sim = covid_seattle.Sim()
to_profile = 'sim' # Must be one of the options listed below

func_options = {'sim':    sim.run,
                # 'update': sc.odict(sim.people)[0].update,
                # 'preg':   sc.odict(sim.people)[0].get_preg_prob,
                # 'get_method': sc.odict(sim.people)[0].get_method,
                }

sc.profile(run=sim.run, follow=func_options[to_profile])
