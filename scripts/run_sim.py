'''
Simple script for running the Covid-19 agent-based model
'''

import covid_abm

do_plot = 1
do_save = 0
verbose = 0

sim = covid_abm.Sim()
sim.set_seed()
sim.run(verbose=verbose)
sim.likelihood()
if do_plot:
    sim.plot(do_save=do_save)