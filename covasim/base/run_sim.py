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
do_show = 1
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
dt1 = sim.get_doubling_time(interval=[3,20])
dt2 = sim.get_doubling_time(start_day=3,end_day=20)
dt3 = sim.get_doubling_time(start_day=3, end_day=20, moving_window=4)
dt4 = sim.get_doubling_time(interval=[3,20], moving_window=6)
dt5 = sim.get_doubling_time(interval=[3,20], moving_window=6, exp_approx=True)
import numpy as np
dt6 = sim.get_doubling_time(series=np.power(1.03, range(100)), interval=[3,30], moving_window=3) # Should be a series with values = 23.44977..
dt7 = sim.get_doubling_time(series=np.power(1.03, range(100)), interval=[3,30], moving_window=3, exp_approx=True) # Should be a series with values = 23.44977..
dt8 = sim.get_doubling_time(start_day=9, end_day=20, moving_window=1, series="cum_recoveries") # Should recast window to 2 then return a series with 100s in it
dt9 = sim.get_doubling_time(start_day=3, end_day=20, moving_window=4, series="cum_deaths") # Should fail, no growth in deaths



