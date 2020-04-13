'''
Simplest possible Covasim usage example.
'''

debug = True

import covasim as cv
if debug:
    import warnings
    warnings.simplefilter("error")

sim = cv.Sim(pop_size=100000, pop_infected=50, n_days=180)
sim.run()
sim.plot()