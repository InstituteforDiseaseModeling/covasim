'''
Simple script for running Covasim scenarios
'''


import sciris as sc
import covasim as cova


if __name__ == "__main__":

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
    do_show = 1
    plot_health = 1
    keep_sims = 0 # WARNING, huge! (>100 MB)
    verbose = 1

    # Sim options
    interv_day = 10
    interv_eff = 0.7

    metapars = dict(
        n_runs = 1, # Number of parallel runs; change to 3 for quick, 11 for real
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
                      'interventions': cova.ChangeBeta(days=interv_day, changes=interv_eff)
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
        fig1 = scens.plot(do_show=do_show)

    if plot_health:
        fig2 = scens.plot_healthsystem()

    sc.toc()

