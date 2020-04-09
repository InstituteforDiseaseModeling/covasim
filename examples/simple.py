'''
Simplest possible Covasim usage example.
'''

import covasim as cv
sim = cv.Sim(pop_size=50000, rescale=1, pop_scale=1000, n_days=180, rescale_factor=10)
sim.run()
sim.plot()