'''
Simple script for running the Covid-19 agent-based model
'''

import pylab as pl
import sciris as sc
import covid_seattle
from covid_abm import utils as cov_ut

do_save = 1
verbose = 0
n = 16
xmin = 0
xmax = 56
noise = 0.2
seed = 1
reskeys = ['cum_exposed', 'cum_deaths']

orig_sim = covid_seattle.Sim()
orig_sim.set_seed(seed)
finished_sims = covid_seattle.multi_run(orig_sim, n=n, noise=noise)

res0 = finished_sims[0].results
npts = len(res0[reskeys[0]])
tvec = res0['t'] + xmin

both = {}
for key in reskeys:
    both[key] = pl.zeros((npts, n))
for key in reskeys:
    for s,sim in enumerate(finished_sims):
        both[key][:,s] = sim.results[key]

best = {}
low = {}
high = {}
for key in reskeys:
    best[key] = both[key].mean(axis=1)*orig_sim['scale']
    low[key] = both[key].min(axis=1)*orig_sim['scale']
    high[key] = both[key].max(axis=1)*orig_sim['scale']

scenarios = {
    'Orig': 'Original COVID-ABM parameters',
}

final = sc.objdict()
final['transGenEpiParams'] = sc.objdict({'best':sc.dcp(best), 'low':sc.dcp(low), 'high':sc.dcp(high)})

for scenkey,scenname in scenarios.items():

    scen_sim = covid_seattle.Sim()

    if scenkey == 'Orig':
        opars = covid_seattle.make_pars_orig()
        scen_sim.pars.update(opars)


    scen_sims = covid_seattle.multi_run(scen_sim, n=n, noise=noise)

    scenboth = {}
    for key in reskeys:
        scenboth[key] = pl.zeros((npts, n))
        for s,sim in enumerate(scen_sims):
            scenboth[key][:,s] = sim.results[key]

    scen_best = {}
    scen_low = {}
    scen_high = {}
    for key in reskeys:
        scen_best[key] = scenboth[key].mean(axis=1)*orig_sim['scale']
        scen_low[key] = scenboth[key].min(axis=1)*orig_sim['scale']
        scen_high[key] = scenboth[key].max(axis=1)*orig_sim['scale']


    final[scenkey] = sc.objdict({'best':sc.dcp(scen_best), 'low':sc.dcp(scen_low), 'high':sc.dcp(scen_high)})


    #%% Plotting

    fig_args     = {'figsize':(20,20)}
    plot_args    = {'lw':3, 'alpha':0.7}
    scatter_args = {'s':150, 'marker':'s'}
    axis_args    = {'left':0.1, 'bottom':0.05, 'right':0.9, 'top':0.97, 'wspace':0.2, 'hspace':0.25}
    fill_args    = {'alpha': 0.3}
    font_size = 18

    fig = pl.figure(**fig_args)
    pl.subplots_adjust(**axis_args)
    pl.rcParams['font.size'] = font_size

    for k,key in enumerate(reskeys):
        pl.subplot(2,1,k+1)

        pl.fill_between(tvec, low[key], high[key], **fill_args)
        pl.fill_between(tvec, scen_low[key], scen_high[key], **fill_args)
        pl.plot(tvec, best[key], label='transGenEpi-like parameters', **plot_args)
        pl.plot(tvec, scen_best[key], label=scenname, **plot_args)

        pl.grid(True)
        cov_ut.fixaxis(sim)
        if k == 0:
            pl.ylabel('Cumulative infections')
        else:
            pl.ylabel('Cumulative deaths')
        pl.xlabel('Days since March 9th')
        pl.xlim([xmin, xmax])
        pl.gca().set_xticks(pl.arange(xmin,xmax+1, 7))
        sc.commaticks(axis='y')

    if do_save:
        pl.savefig(f'seattle_projections_v2_{scenkey}.png')


#%% Print statistics
for k in ['transGenEpiParams'] + list(scenarios.keys()):
    for key in reskeys:
        print(f'{k} {key}: {final[k].best[key][-1]:0.0f}')

sc.saveobj('seattle-projection-results_v2.obj', final)
