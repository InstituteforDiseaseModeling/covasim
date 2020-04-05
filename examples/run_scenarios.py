'''
Simple script for running Covasim scenarios
'''

import sciris as sc
import covasim as cv


sc.heading('Setting up...')

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
do_show = 1
keep_sims = 0 # WARNING, huge! (>100 MB)
verbose = 1

# Sim options
interv_day = 35
interv_eff = 0.7
default_beta = 0.015 # Should match parameters.py

basepars = dict(
  n = 5000
)

metapars = dict(
    n_runs = 3, # Number of parallel runs; change to 3 for quick, 11 for real
    noise = 0.1, # Use noise, optionally
    noisepar = 'beta',
    seed = 1,
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
                  'interventions': None,
                  }
              },
            'distance': {
              'name':'Social distancing',
              'pars': {
                  'interventions': cv.change_beta(days=interv_day, changes=interv_eff)
                  }
              },
            # 'distance2': { # With noise = 0.0, this should be identical to the above
            #   'name':'Social distancing, version 2',
            #   'pars': {
            #       'interventions': cv.dynamic_pars({'beta':dict(days=interv_day, vals=interv_eff*default_beta)})
            #       }
            #   },
             }


if __name__ == "__main__": # Required for parallel processing on Windows

    sc.tic()

    # If we're rerunning...
    if do_run:
        scens = cv.Scenarios(basepars=basepars, metapars=metapars, scenarios=scenarios)
        scens.run(verbose=verbose)
        if do_save:
            scens.save(filename=obj_path, keep_sims=keep_sims)

    # Don't run
    else:
        scens = cv.Scenarios.load(obj_path)

    if do_plot:
        fig1 = scens.plot(do_show=do_show)

    sc.toc()

