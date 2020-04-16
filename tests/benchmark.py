# Benchmark the simulation

import sciris as sc
import covasim as cv

sim = cv.Sim()
sim['n_days'] = 180
to_profile = 'step' # Must be one of the options listed below...currently only 1

func_options = {
    'person':      cv.Person.__init__,
    'make_people': cv.make_people,
    'init_people': sim.init_people,
    'initialize':  sim.initialize,
    'run':         sim.run,
    'step':        sim.step,
}

sc.profile(run=sim.run, follow=func_options[to_profile])
