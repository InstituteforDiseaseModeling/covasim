'''
Simplest possible Covasim usage example.
'''

import covasim as cv
sim = cv.Sim(pop_infected=5000)
sim.run()
sim.plot()