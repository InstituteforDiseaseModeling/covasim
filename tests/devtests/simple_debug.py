'''
Simplest Covasim usage example.
'''

import covasim as cv

debug = True
if debug:
    import warnings
    warnings.simplefilter("error")

sim = cv.Sim(pop_size=1000, pop_infected=50, n_days=180)
sim.run()
sim.plot()