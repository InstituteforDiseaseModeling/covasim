'''
Simple script for running the Covid-19 agent-based model
'''

import sciris as sc

print('Importing...')
sc.tic()
import covid_seattle
sc.toc()

do_plot = 1
do_save = 0
verbose = 5

print('Making sim...')
sc.tic()
sim = covid_seattle.Sim()
sim.set_seed(1)

print('Running...')
sim.run(verbose=verbose)
if do_plot:
    sim.plot(do_save=do_save)