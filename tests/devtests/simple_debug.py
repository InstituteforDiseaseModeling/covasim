'''
Simplest possible Covasim usage example.
'''

debug = True

import covasim as cv
if debug:
    import warnings
    warnings.simplefilter("error")

sim = cv.Sim(beta=0.001, pop_size=100, pop_infected=50, n_days=90)
sim.run()
sim.plot()