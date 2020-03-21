'''
Rather simple script for running the Covid-19 agent-based model
'''

import covasim as cova
sim = cova.Sim()
sim.run(do_plot=True)