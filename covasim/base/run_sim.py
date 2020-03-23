'''
Simple script for running the Covid-19 agent-based model
'''

import matplotlib
matplotlib.use('TkAgg')
import sciris as sc


print('Importing...')
sc.tic()
import covasim as cova
sc.toc()

do_plot = 1
do_save = 1
do_show = 0
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
sim.set_seed(seed)

print('Running...')
sim.run(verbose=verbose)

if do_plot:
    print('Plotting...')
    fig = sim.plot(do_save=do_save, do_show=do_show, fig_path=fig_path)
