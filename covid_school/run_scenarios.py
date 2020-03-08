'''
Simple script for running the Covid-19 agent-based model
'''

import pylab as pl
import sciris as sc
import covid_school
from covid_abm import utils as cov_ut

do_save = 0
verbose = 0
n = 6
xmin = 9
xmax = 31
key = 'cum_exposed'

orig_sim = covid_school.Sim()
finished_sims = covid_school.multi_run(orig_sim, n=n)

res0 = finished_sims[0].results
npts = len(res0[key])
tvec = res0['t'] + xmin

allres = pl.zeros((npts, n))
for s,sim in enumerate(finished_sims):
    allres[:,s] = sim.results[key]

best = allres.mean(axis=1)*orig_sim['scale']
low = allres.min(axis=1)*orig_sim['scale']
high = allres.max(axis=1)*orig_sim['scale']


#%% Plotting

fig_args     = {'figsize':(20,12)}
plot_args    = {'lw':3, 'alpha':0.7}
scatter_args = {'s':150, 'marker':'s'}
axis_args    = {'left':0.1, 'bottom':0.05, 'right':0.9, 'top':0.97, 'wspace':0.2, 'hspace':0.25}
fill_args    = {'alpha': 0.3}
font_size = 18

fig = pl.figure(**fig_args)
pl.subplots_adjust(**axis_args)
pl.rcParams['font.size'] = font_size
pl.fill_between(tvec, low, high, **fill_args)
pl.plot(tvec, best, label='Business as usual', **plot_args)

pl.grid(True)
cov_ut.fixaxis(sim)
pl.ylabel('Cumulative infections')
pl.xlabel('Date in March')
pl.xlim([xmin, xmax])
pl.gca().set_xticks(pl.arange(xmin,xmax+1))
sc.commaticks(axis='y')
