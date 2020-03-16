'''
Simple script for running the Covid-19 agent-based model
'''

# Standard imports
import pylab as pl
import datetime as dt
import sciris as sc
import covasim.cova_seattle as cova

sc.heading('Setting up...')

## CONFIGURATION

## Read in configuration.  Better would be to extract from simulation output.


pars = cova.make_pars() # TODO: should be gotten from a sim, perhaps

sc.tic()

# Whether or not to run!
do_run = 1
verbose = 0

# Other options
do_save = 0
version = 'v5'
folder = 'results_2020mar15'
fn_fig = f'{folder}/seattle-projections_2020mar15_{version}.png'
fn_obj = f'{folder}/seattle-projection_2020mar15_{version}.obj'

noise = 0*0.2 # Turn off noise
noisepar = 'r_contact'
n = 4 # Number of replicates per configuration
seed = 1


# Plotting configuration
reskeys = ['cum_exposed', 'n_exposed']#, 'cum_deaths']
xmin = pars['day_0']
xmax = xmin + pars['n_days']
errorbars = False
fig_args     = {'figsize':(16,12)}
plot_args    = {'lw':3, 'alpha':0.7}
scatter_args = {'s':150, 'marker':'s'}
axis_args    = {'left':0.10, 'bottom':0.05, 'right':0.95, 'top':0.90, 'wspace':0.5, 'hspace':0.25}
fill_args    = {'alpha': 0.3}
font_size = 18



## Setup a scenarios.  Value will appear in legend.
scenarios = {
    'Baseline': 'Business as usual',
    '25': '25% contact reduction',
    '50': '50% contact reduction',
    '75': '75% contact reduction',
}

## BEGIN REAL WORK
if not do_run:
    # If not running, load from obj
    final = sc.loadobj(fn_obj)
else:
    sc.heading('Baseline run')

    final = sc.objdict()

    npts = None
    # Loop over scenarios and run each in turn
    for scenkey,scenname in scenarios.items():

        scen_sim = cova.Sim()
        scen_sim.set_seed(seed)
        if scenkey == '25':
            scen_sim['quarantine'] = 17
            scen_sim['quarantine_eff'] = 0.75
        elif scenkey == '50':
            scen_sim['quarantine'] = 17
            scen_sim['quarantine_eff'] = 0.50
        elif scenkey == '75':
            scen_sim['quarantine'] = 17
            scen_sim['quarantine_eff'] = 0.25

        # Run
        sc.heading(f'Multirun for {scenkey}')

        scen_sims = cova.multi_run(scen_sim, n=n, noise=noise, noisepar=noisepar)

        sc.heading(f'Processing {scenkey}')

        # From the first simulation, reconstruct the time vector and number of points
        # TODO: Extract time from simulation results
        if npts is None:
            res0 = scen_sims[0].results
            npts = len(res0[reskeys[0]])

        # Store all simulations
        allres = {}
        for key in reskeys:
            allres[key] = pl.zeros((npts, n))
            for s,sim in enumerate(scen_sims):
                allres[key][:,s] = sim.results[key]

        # Keep best, low, and high for errorbars
        scen_best = {}
        scen_low = {}
        scen_high = {}
        for key in reskeys:
            scen_best[key] = pl.median(allres[key], axis=1)
            scen_low[key] = allres[key].min(axis=1)
            scen_high[key] = allres[key].max(axis=1)

        final[scenkey] = sc.objdict({'scenname': scenname, 'best':sc.dcp(scen_best), 'low':sc.dcp(scen_low), 'high':sc.dcp(scen_high)})

## NOW FOR PLOTTING
sc.heading('Plotting')

fig = pl.figure(**fig_args)
pl.subplots_adjust(**axis_args)
pl.rcParams['font.size'] = font_size
pl.rcParams['font.family'] = 'Proxima Nova'

# Create the tvec based on the results
tvec = xmin+pl.arange(len(final['Baseline']['best'][reskeys[0]]))


#%% Plotting
for k,key in enumerate(reskeys):
    pl.subplot(len(reskeys),1,k+1, label=key)

    for fkey, data in final.items():
        scenname = scenarios[fkey]
        if errorbars:
            pl.fill_between(tvec, data['low'][key], data['high'][key], **fill_args)
        pl.plot(tvec, data['best'][key], label=scenname, **plot_args)

        interv_col = [0.5, 0.2, 0.4]

        if key == 'cum_exposed':
            sc.setylim()
            pl.title('Cumulative infections')
            pl.legend()
            pl.text(xmin+16.5, 12000, 'Intervention', color=interv_col, fontstyle='italic')

            pl.text(xmin-6, 30e3, 'COVID-19 projections, King + Snohomish counties', fontsize=24)

        elif key == 'n_exposed':
            sc.setylim()
            pl.title('Active infections')

        pl.grid(True)

        pl.plot([xmin+16]*2, pl.ylim(), '--', lw=2, c=interv_col) # Plot intervention
        pl.gca().set_xticks(pl.arange(xmin, xmax+1, 7))

        # Put calendar time on the x-axis
        xt = pl.gca().get_xticks()
        lab = []
        for t in xt:
            tmp = dt.datetime(2020, 1, 1) + dt.timedelta(days=int(t))
            lab.append( tmp.strftime('%B %d') )
        pl.gca().set_xticklabels(lab)

        sc.commaticks(axis='y')


#%% Print statistics
for k in list(scenarios.keys()):
    for key in reskeys:
        print(f'{k} {key}: {final[k].best[key][-1]:0.0f}')

# Save if requested
if do_save:
    pl.savefig(fn_fig, dpi=150)
    if do_run: # Don't resave loaded data
        sc.saveobj(fn_obj, final)

sc.toc()
pl.show()

