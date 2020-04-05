'''
Simple script for running the Covid-19 agent-based model
'''

import sciris as sc
import os
import json

print('Importing...')
sc.tic()
import covasim as cv
sc.toc()

do_plot = 1
do_save = 1
do_show = 0
do_summary = 1
verbose = 1
seed    = 4
interv  = 0

version  = 'v0'
date     = '2020mar31'
folder   = 'results'
basename = f'{folder}/covasim_run_{date}_{version}'
fig_path = f'{basename}.png'
summary_path = f'{basename}.json'

print('Making sim...')
sc.tic()
sim = cv.Sim()
sim.set_seed(seed) # Set seed (can also be done via sim['seed'] = seed)
sim['n'] = 5000 # Population size
sim['n_days'] = 180 # Number of days to simulate
sim['prog_by_age'] = True # Use age-specific mortality etc.
sim['usepopdata'] = False # Use realistic population structure (requires synthpops)
# sim['rel_death_prob'] = 0.0
if interv:
    sim['interventions'] = cv.change_beta(days=45, changes=0.5) # Optionally add an intervention

print('Running...')
sim.run(verbose=verbose)

if do_plot || do_summary:
    # Clean the results dir
    if os.path.exists(folder):
        os.rmdir(folder)
    os.mkdir(folder)

if do_plot:
    print('Plotting...')
    fig = sim.plot(do_save=do_save, do_show=do_show, fig_path=fig_path)

if do_summary:
    summary = sim.summary_stats(verbose=1)
    f = open(summary_path, "w")
    f.write(json.dumps(summary, indent=2, separators=(',', ': ')))
    f.close()
