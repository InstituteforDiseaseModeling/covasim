'''
Simplest possible Covasim usage example.
'''

debug = True

import covasim as cv
if debug:
    import warnings
    warnings.simplefilter("error")

sim = cv.Sim(pop_size=1000, pop_infected=10, n_days=60)
sim.run()
sim.plot()