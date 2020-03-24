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

do_plot = 0
do_save = 0
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


# Test doubling time
dt1 = sim.get_doubling_time(interval=[3,sim['n_days']+10], verbose=2) # should reset end date to sim['n_days'] and return 5.681208944
dt2 = sim.get_doubling_time(start_day=3,end_day=sim['n_days']) # should return 5.681208944
dt3 = sim.get_doubling_time(interval=[3,sim['n_days']], exp_approx=True) # should return 5.2679
dt4 = sim.get_doubling_time(start_day=3, end_day=sim['n_days'], moving_window=4) # should return array
#import numpy as np
dt5 = sim.get_doubling_time(series=np.power(1.03, range(100)), interval=[3,30], moving_window=3) # Should be a series with values = 23.44977..
dt6 = sim.get_doubling_time(start_day=9, end_day=20, moving_window=1, series="cum_recoveries") # Should recast window to 2 then return a series with 100s in it
dt7 = sim.get_doubling_time(start_day=3, end_day=20, moving_window=4, series="cum_deaths") # Should fail, no growth in deaths



