'''
Simple script for running the Covid-19 agent-based model
'''

import sciris as sc

print('Importing...')
sc.tic()
import covasim.cova_oregon as cova
sc.toc()

do_plot = 1
do_save = 0
verbose = 1
seed = 1
folder = 'results_2020mar14/'
fig_fn =  folder + 'oregon-projection-calibration_v0.png'

print('Making sim...')
sc.tic()
sim = cova.Sim()
sim.set_seed(seed)

print('Running...')
sim.run(verbose=verbose)
if do_plot:
    sim.plot(do_save=fig_fn)