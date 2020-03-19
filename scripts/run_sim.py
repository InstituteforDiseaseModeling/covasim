'''
Simple script for running the Covid-19 agent-based model
'''

import covasim.cova_generic as cova

do_plot = 1
verbose = 1

sim = cova.Sim()
sim.run(verbose=verbose)
if do_plot:
    sim.plot()
