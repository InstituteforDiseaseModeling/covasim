'''
Rather simple script for running the Covid-19 agent-based model
'''

import covasim as cv
sim = cv.Sim()
sim.run()
sim.plot()