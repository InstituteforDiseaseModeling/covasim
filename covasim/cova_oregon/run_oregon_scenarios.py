'''
Simple script for running the Covid-19 agent-based model
'''

import pylab as pl
import datetime as dt
import sciris as sc
import covasim.cova_oregon as cova

sc.heading('Setting up...')

sc.tic()

# Whether or not to run!
do_run = 1

# Other options
do_save = 1
verbose = 1
n = 12
xmin = 52 # pars['day_0']
xmax = xmin+50 # xmin + pars['n_days']
interv_day = 24
closure_len = 14
noise = 0.25 # Use noise, optionally
noisepar = 'beta'
seed = 1
reskeys = ['cum_exposed', 'n_exposed']
quantiles = {'low':0.0, 'high':0.9}

version = 'v2'
folder = 'results_2020mar14'
fn_fig = f'{folder}/oregon-projections_2020mar14_{version}.png'
fn_obj = f'{folder}/oregon-projection-2020mar14_{version}.obj'


scenarios = {
    'baseline':   'Business as usual',
    'reopen':     'Current interventions, schools reopen',
    'closed':     'Current interventions, schools stay closed',
    'aggressive': 'Aggressive interventions (business closures)',
}

# If we're rerunning...
if do_run:

    final = sc.objdict()

    for scenkey,scenname in scenarios.items():

        scen_sim = cova.Sim()
        scen_sim.set_seed(seed)
        if scenkey == 'baseline':
            scen_sim['interv_days'] = [] # No interventions
            scen_sim['interv_effs'] = []
        elif scenkey == 'reopen':
            scen_sim['interv_days'] = [interv_day, interv_day+closure_len] # Close schools for 2 weeks starting Mar. 16, then reopen
            scen_sim['interv_effs'] = [0.5, 1.6]
        elif scenkey == 'closed':
            scen_sim['interv_days'] = [interv_day] # Close schools for 2 weeks starting Mar. 16, then reopen
            scen_sim['interv_effs'] = [0.5]
        elif scenkey == 'aggressive':
            scen_sim['interv_days'] = [interv_day] # Close everything
            scen_sim['interv_effs'] = [0.1]

        sc.heading(f'Multirun for {scenkey}')

        scen_sims = cova.multi_run(scen_sim, n=n, noise=noise, noisepar=noisepar, verbose=verbose)

        sc.heading(f'Processing {scenkey}')

        # TODO: this only needs to be done once and can be done so much better!
        res0 = scen_sims[0].results
        npts = len(res0[reskeys[0]])
        tvec = xmin+res0['t']

        scenboth = {}
        for key in reskeys:
            scenboth[key] = pl.zeros((npts, n))
            for s,sim in enumerate(scen_sims):
                scenboth[key][:,s] = sim.results[key]

        scen_best = {}
        scen_low = {}
        scen_high = {}
        for key in reskeys:
            scen_best[key] = pl.median(scenboth[key], axis=1)
            scen_low[key]  = pl.quantile(scenboth[key], q=quantiles['low'], axis=1)
            scen_high[key] = pl.quantile(scenboth[key], q=quantiles['high'], axis=1)



        final[scenkey] = sc.objdict({'scenname': scenname, 'best':sc.dcp(scen_best), 'low':sc.dcp(scen_low), 'high':sc.dcp(scen_high)})

# Don't run
else:
    final = sc.loadobj(fn_obj)

sc.heading('Plotting')

fig_args     = {'figsize':(16,12)}
plot_args    = {'lw':3, 'alpha':0.7}
scatter_args = {'s':150, 'marker':'s'}
axis_args    = {'left':0.10, 'bottom':0.05, 'right':0.95, 'top':0.90, 'wspace':0.5, 'hspace':0.25}
fill_args    = {'alpha': 0.2}
font_size = 18
fig = pl.figure(**fig_args)
pl.subplots_adjust(**axis_args)
pl.rcParams['font.size'] = font_size
pl.rcParams['font.family'] = 'Proxima Nova'

# Create the tvec based on the results -- #TODO: make better!
tvec = xmin+pl.arange(len(final['baseline']['best'][reskeys[0]]))




#%% Plotting
for k,key in enumerate(reskeys):
    pl.subplot(len(reskeys),1,k+1)

    for datakey, data in final.items():
        print(datakey)
        if datakey in scenarios:
            scenname = scenarios[datakey]
        else:
            scenname = 'Business as usual'

        pl.fill_between(tvec, data['low'][key], data['high'][key], **fill_args)
        pl.plot(tvec, data['best'][key], label=scenname, **plot_args)

        interv_col = [0.5, 0.2, 0.4]

        ymax = pl.ylim()[1]

        if key == 'cum_exposed':
            sc.setylim()
            pl.title('Cumulative infections')
            pl.legend()
            pl.text(xmin+interv_day+0.5, ymax*0.85, 'Interventions\nbegin', color=interv_col, fontstyle='italic')
            pl.text(xmin+interv_day+closure_len-5, ymax*0.8, 'Proposed\nreopening\nof schools', color=interv_col, fontstyle='italic')

            pl.text(0.0, 1.1, 'COVID-19 projections, Oregon', fontsize=24, transform=pl.gca().transAxes)

        elif key == 'n_exposed':
            sc.setylim()
            pl.title('Active infections')

        pl.grid(True)

        pl.plot([xmin+interv_day]*2, pl.ylim(), '-', lw=1, c=interv_col) # Plot intervention
        pl.plot([xmin+interv_day+closure_len]*2, pl.ylim(), '-', lw=1, c=interv_col) # Plot intervention
        # pl.xlabel('Date')
        # pl.ylabel('Count')

        # Set x-axis
        pl.gca().set_xticks(pl.arange(xmin, xmax+1, 7))
        xt = pl.gca().get_xticks()
        lab = []
        for t in xt:
            tmp = dt.datetime(2020, 1, 1) + dt.timedelta(days=int(t)) # + pars['day_0']
            lab.append( tmp.strftime('%B %d') )
        pl.gca().set_xticklabels(lab)
        sc.commaticks(axis='y')

if do_save:
    pl.savefig(fn_fig)


#%% Print statistics
for k in list(scenarios.keys()):
    for key in reskeys:
        print(f'{k} {key}: {final[k].best[key][-1]:0.0f}')

if do_save:
    pl.savefig(fn_fig, dpi=150)
    if do_run: # Don't resave loaded data
        sc.saveobj(fn_obj, final)

sc.toc()
pl.show()

