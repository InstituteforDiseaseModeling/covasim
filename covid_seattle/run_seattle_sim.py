'''
Simple script for running the Covid-19 agent-based model
'''

import covid_seattle

do_plot = 1
do_save = 0
verbose = 1

sim = covid_seattle.Sim()
sim.set_seed(1) # 4 ok, 5 ok, 6 good
sim.run(verbose=verbose)
if do_plot:
    sim.plot(do_save=do_save)