'''
Simple script for running Covasim scenarios
'''


import sciris as sc
import covasim as cova


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
keep_sims = 0 # WARNING, huge! (>100 MB)
verbose = 1

# Sim options
interv_day = 10
closure_len = 14

metapars = dict(
    n = 3, # Number of parallel runs; change to 3 for quick, 11 for real
    noise = 0.1, # Use noise, optionally
    noisepar = 'beta',
    seed = 1,
    reskeys = ['cum_exposed', 'n_exposed'],
    quantiles = {'low':0.1, 'high':0.9},
)

# For saving
version  = 'v0'
date     = '2020mar24'
folder   = 'results'
basename = f'{folder}/covasim_scenarios_{date}_{version}'
fig_path   = f'{basename}.png'
obj_path   = f'{basename}.scens'

# Define the scenarios
scenarios = {'baseline': {
              'name':'Baseline',
              'pars': {
                  'interv_days': [],
                  'interv_effs': [],
                  }
              },
            'distance': {
              'name':'Social distancing',
              'pars': {
                  'interv_days': [interv_day],
                  'interv_effs': [0.7],
                  }
              },
             }


# If we're rerunning...
if do_run:
    scens = cova.Scenarios(metapars=metapars, scenarios=scenarios)
    scens.run(keep_sims=keep_sims, verbose=verbose)
    if do_save:
        scens.save(filename=obj_path)

# Don't run
else:
    scens = cova.Scenarios.load(obj_path)


if do_plot:
    scens.plot(show_plot=show_plot)




sc.toc()

