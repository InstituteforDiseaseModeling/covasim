'''
Simple script for running the Covid-19 agent-based model
'''

import sciris as sc
import pylab as pl
import datetime as dt

print('Importing...')
sc.tic()
import covasim.cova_oregon as cova
sc.toc()

do_plot = 1
do_save = 1
verbose = 1
just_calib = 1 # Just show the calibration period
seed = 1
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
    fig.set_size_inches((16,12))

    # Set x-axis
    for ax in fig.axes:
        xmin,xmax = ax.get_xlim()
        ax.set_xticks(pl.arange(xmin, xmax+1, 7))
        xt = ax.get_xticks()
        print(xt)
        lab = []
        for t in xt:
            tmp = sim['day_0'] + dt.timedelta(days=int(t)) # + pars['day_0']
            lab.append(tmp.strftime('%B %d'))
        ax.set_xticklabels(lab)
        ax.set_xlabel(None)
        sc.commaticks(axis='y')

    pl.savefig(fig_fn)