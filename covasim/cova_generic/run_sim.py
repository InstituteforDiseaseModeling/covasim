'''
Simple script for running the Covid-19 agent-based model
'''

import matplotlib
matplotlib.use('TkAgg')
import sciris as sc

print('Importing...')
sc.tic()
import covasim.cova_generic as cova
sc.toc()

do_plot = 0
do_save = 1
verbose = 2
seed = 1

version  = 'v0'
date     = '2020mar18'
folder   = 'results'
basename = f'{folder}/covasim_run_{date}_{version}'
fig_path   = f'{basename}.png'

print('Making sim...')
sc.tic()
sim = cova.Sim()
sim.set_seed(seed)


# Print extras

print('Running...')
sim.run(verbose=verbose)
if do_plot:
    fig = sim.plot(do_save=do_save, fig_path=fig_path)
