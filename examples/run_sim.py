'''
Simple script for running the Covid-19 agent-based model
'''

import sciris as sc

print('Importing...')
sc.tic()
import covasim as cv
sc.toc()

do_plot = 1
do_save = 0
do_show = 1
verbose = 1
seed    = 4
interv  = 1

version  = 'v0'
date     = '2020mar21'
folder   = 'results'
basename = f'{folder}/covasim_run_{date}_{version}'
fig_path = f'{basename}.png'

print('Making sim...')
sc.tic()
sim = cv.Sim()
sim.set_seed(seed)
if interv:
    sim['interventions'] = cv.change_beta(days=45, changes=0.5)

print('Running...')
sim.run(verbose=verbose)

if do_plot:
    print('Plotting...')
    fig = sim.plot(do_save=do_save, do_show=do_show, fig_path=fig_path)










