'''
Simple script for running Covasim scenarios
'''

import pylab as pl
import datetime as dt
import sciris as sc
import covasim as cova
import covid_healthsystems as covidhs


sc.heading('Setting up...')

sc.tic()

# Specify what to run
scenarios = {
    'baseline':     'Status quo',
    'distance':    'Social distancing',
    # 'isolatepos':   'Isolate people who diagnose positive',
}

# Run options
do_run = 1
do_save = 0 # refers to whether to save plot - see also save_sims
do_plot = 1
show_plot = 1
save_sims = 0 # WARNING, huge! (>100 MB)
verbose = 1
n = 3 # Number of parallel runs; change to 3 for quick, 11 for real

# Sim options
interv_day = 10
closure_len = 14
noise = 0.1 # Use noise, optionally
noisepar = 'beta'
seed = 1
reskeys = ['cum_exposed', 'n_exposed']
quantiles = {'low':0.1, 'high':0.9}

# For saving
version  = 'v0'
date     = '2020mar24'
folder   = 'results'
basename = f'{folder}/covasim_scenarios_{date}_{version}'
fig_path   = f'{basename}.png'
obj_path   = f'{basename}.obj'



# If we're rerunning...
if do_run:

    # Order is: results key, scenario, best/low/high
    allres = sc.objdict()
    for reskey in reskeys:
        allres[reskey] = sc.objdict()
        for scenkey in scenarios.keys():
            allres[reskey][scenkey] = sc.objdict()
            for nblh in ['name', 'best', 'low', 'high']:
                allres[reskey][scenkey][nblh] = None # This will get populated below

    for scenkey,scenname in scenarios.items():

        scen_sim = cova.Sim()
        scen_sim.set_seed(seed)

        if scenkey == 'baseline':
            scen_sim['interv_days'] = [] # No interventions
            scen_sim['interv_effs'] = []

        elif scenkey == 'distance':
            scen_sim['interv_days'] = [interv_day] # Close schools for 2 weeks starting Mar. 16, then reopen
            scen_sim['interv_effs'] = [0.7] # Change to 40% and then back to 70%

        elif scenkey == 'isolatepos':
            scen_sim['diag_factor'] = 0.1 # Scale beta by this amount for anyone who's diagnosed

        else:
            raise KeyError


        sc.heading(f'Multirun for {scenkey}')

        scen_sims = cova.multi_run(scen_sim, n=n, noise=noise, noisepar=noisepar, verbose=verbose)

        sc.heading(f'Processing {scenkey}')

        # TODO: this only needs to be done once
        res0 = scen_sims[0].results
        npts = res0[reskeys[0]].npts
        tvec = res0['t']

        scenraw = {}
        for reskey in reskeys:
            scenraw[reskey] = pl.zeros((npts, n))
            for s,sim in enumerate(scen_sims):
                scenraw[reskey][:,s] = sim.results[reskey].values

        scenres = sc.objdict()
        scenres.best = {}
        scenres.low = {}
        scenres.high = {}
        for reskey in reskeys:
            scenres.best[reskey] = pl.mean(scenraw[reskey], axis=1) # Changed from median to mean for smoother plots
            scenres.low[reskey]  = pl.quantile(scenraw[reskey], q=quantiles['low'], axis=1)
            scenres.high[reskey] = pl.quantile(scenraw[reskey], q=quantiles['high'], axis=1)

        for reskey in reskeys:
            allres[reskey][scenkey]['name'] = scenname
            for blh in ['best', 'low', 'high']:
                allres[reskey][scenkey][blh] = scenres[blh][reskey]

        if save_sims:
            print('WARNING: saving sims, which will produce a very large file!')
            allres['sims'] = scen_sims

# Don't run
else:
    simset = sc.loadobj(obj_path)


if do_plot:
    simset.plot()


#%% Print statistics
for reskey in reskeys:
    for scenkey in list(scenarios.keys()):
        print(f'{reskey} {scenkey}: {allres[reskey][scenkey].best[-1]:0.0f}')


# Perform health systems analysis
hsys = covidhs.HealthSystem(allres)
hsys.analyze()
hsys.plot()

sc.toc()

