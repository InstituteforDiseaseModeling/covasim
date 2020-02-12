'''
Simple script for running the Covid-19 agent-based model
'''

import covid_abm

do_plot = True
do_save = False

sim = covid_abm.Sim()
sim.run()
if do_plot:
    sim.plot(do_save=do_save)