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
folder = 'results_2020mar14/'
fig_fn =  folder + 'oregon-projection-calibration_v1.png'

print('Making sim...')
sc.tic()
sim = cova.Sim()
sim.set_seed(seed)
if just_calib:
    sim['n_days'] = 21


print('Running...')
sim.run(verbose=verbose)
if do_plot:
    fig = sim.plot(do_save=False)
    if do_save:
        pl.savefig(fig_fn)