'''
Simplest possible Covasim usage example.
'''

import covasim as cv
sim = cv.Sim(rescale=1, pop_scale=1000, n_days=180)
sim.run()
sim.plot()