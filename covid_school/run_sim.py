'''
Simple script for running the Covid-19 agent-based model
'''

import covid_school

do_plot = 1
do_save = 0
verbose = 0

sim = covid_school.Sim()
sim.set_seed(5) # 4 ok, 5 ok, 6 good
sim.run(verbose=verbose)
sim.likelihood()
if do_plot:
    sim.plot(do_save=do_save)