'''
Simplest possible Covasim usage example.
'''

import covasim as cv
sim = cv.Sim(pop_size=1000, pop_infected=1)
sim.run()
sim.plot()