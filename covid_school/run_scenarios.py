'''
Simple script for running the Covid-19 agent-based model
'''

import pylab as pl
import sciris as sc
import covid_school
from covid_abm import utils as cov_ut

do_save = 1
verbose = 0
n = 6
xmin = 9
xmax = 31
noise = 0.2
key = 'cum_exposed'

orig_sim = covid_school.Sim()
finished_sims = covid_school.multi_run(orig_sim, n=n, noise=noise)

res0 = finished_sims[0].results
npts = len(res0[key])
tvec = res0['t'] + xmin

allres = pl.zeros((npts, n))
for s,sim in enumerate(finished_sims):
    allres[:,s] = sim.results[key]

best = allres.mean(axis=1)*orig_sim['scale']
low = allres.min(axis=1)*orig_sim['scale']
high = allres.max(axis=1)*orig_sim['scale']

scenarios = {
    # 'del50': 'Closure in 1 week, 50% reduction', 
    # 'im50': 'Immediate closure, 50% reduction', 
    # 'im16': 'Immediate closure, 16% reduction',
    # 'im95': 'Immediate closure, 95% reduction',
    'short': 'One-week closure, 50% reduction',
}

final = sc.objdict()
final['baseline'] = sc.objdict({'best':sc.dcp(best), 'low':sc.dcp(low), 'high':sc.dcp(high)})

for scenkey,scenname in scenarios.items():
    
    scen_sim = covid_school.Sim()
    if scenkey == 'del50':
        scen_sim['quarantine'] = 7
        scen_sim['quarantine_eff'] = 0.5
    elif scenkey == 'im50':
        scen_sim['quarantine'] = 0
        scen_sim['quarantine_eff'] = 0.5
    elif scenkey == 'im16':
        scen_sim['quarantine'] = 0
        scen_sim['quarantine_eff'] = 0.84
    elif scenkey == 'im95':
        scen_sim['quarantine'] = 0
        scen_sim['quarantine_eff'] = 0.05
    elif scenkey == 'short':
        scen_sim['quarantine'] = 0
        scen_sim['unquarantine'] = 7
        scen_sim['quarantine_eff'] = 0.50
    
        
    scen_sims = covid_school.multi_run(scen_sim, n=n, noise=noise)
    
    scenres = pl.zeros((npts, n))
    for s,sim in enumerate(scen_sims):
        scenres[:,s] = sim.results[key]
    
    scen_best = scenres.mean(axis=1)*orig_sim['scale']
    scen_low = scenres.min(axis=1)*orig_sim['scale']
    scen_high = scenres.max(axis=1)*orig_sim['scale']
    
    final[scenkey] = sc.objdict({'best':sc.dcp(scen_best), 'low':sc.dcp(scen_low), 'high':sc.dcp(scen_high)})


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
    pl.fill_between(tvec, scen_low, scen_high, **fill_args)
    pl.plot(tvec, best, label='Business as usual', **plot_args)
    pl.plot(tvec, scen_best, label=scenname, **plot_args)
    
    pl.grid(True)
    cov_ut.fixaxis(sim)
    pl.ylabel('Cumulative infections')
    pl.xlabel('Date in March')
    pl.xlim([xmin, xmax])
    pl.gca().set_xticks(pl.arange(xmin,xmax+1))
    sc.commaticks(axis='y')
    
    if do_save:
        pl.savefig(f'school_closure_{scenkey}.png')


#%% Print statistics
for k in ['baseline'] + list(scenarios.keys()):
    print(f'{k}: {final[k].best[-1]:0.0f}')

sc.saveobj('school-results.obj', final)
