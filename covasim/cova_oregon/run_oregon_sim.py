'''
Simple script for running the Covid-19 agent-based model
'''

import sciris as sc
import pylab as pl

print('Importing...')
sc.tic()
import covasim.cova_oregon as cova
sc.toc()

do_plot = 1
do_save = 1
verbose = 1
just_calib = 1 # Just show the calibration period
seed = 1 # 1092843

version = 'v1'
date     = '2020mar16'
folder   = f'results_{date}'
fig_fn =  f'{folder}/oregon-calibration_{version}.png'

print('Making sim...')
sc.tic()
sim = cova.Sim()
sim.set_seed(seed)
if just_calib:
    sim['n_days'] = 28


print('Running...')
sim.run(verbose=verbose)
if do_plot:
    fig = sim.plot(do_save=False)
    if do_save:
        pl.savefig(fig_fn)