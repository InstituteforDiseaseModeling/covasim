'''
Simplest possible Covasim usage example.
'''

import covasim as cv
sim = cv.Sim(beta=0.015)
sim.run()
sim.plot()