'''
Simple script for running the Covid-19 agent-based model
'''

import pylab as pl
import sciris as sc

print('Importing...')
sc.tic()
import covasim.cova_cdc as cova
sc.toc()

do_plot = 1
do_save = 1
verbose = 1
seed = 1
folder = 'results_2020mar15'
version = 'v1'
fig_fn =  f'{folder}/cdc-projection_{version}.png'

print('Making sim...')
sc.tic()
sim = cova.Sim()
sim.set_seed(seed)

print('Running...')
sim.run(verbose=verbose)
if do_plot:
    fig = sim.plot(do_save=False)
    if do_save:
        pl.savefig(fig_fn)
