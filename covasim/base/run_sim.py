'''
Simple script for running the Covid-19 agent-based model
'''

import sciris as sc

print('Importing...')
sc.tic()
import covasim as cova
sc.toc()

do_plot = 1
do_save = 0
verbose = 1
seed    = 1

version  = 'v0'
date     = '2020mar21'
folder   = 'results'
basename = f'{folder}/covasim_run_{date}_{version}'
fig_path = f'{basename}.png'

print('Making sim...')
sc.tic()
sim = cova.Sim()
sim.set_seed(None)

print('Running...')
sim.run(verbose=verbose)
if do_plot:
    fig = sim.plot(do_save=do_save, fig_path=fig_path)
