'''
Simple script for running the Covid-19 agent-based model
'''

import covid_school

do_plot = 1
do_save = 0
verbose = 0

sim1 = covid_school.Sim()
sim.set_seed(1)
sim1.run(verbose=verbose)
if do_plot:
    sim.plot(do_save=do_save)