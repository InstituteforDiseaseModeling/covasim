'''
Simple script for running Covasim scenarios
'''

import covasim as cv

# Run options
do_plot = 1
do_show = 1
verbose = 0

# Sim options
basepars = dict(
  pop_size = 2000,
  verbose = verbose,
)

metapars = dict(
    n_runs    = 3, # Number of parallel runs; change to 3 for quick, 11 for real
    noise     = 0.1, # Use noise, optionally
    noisepar  = 'beta',
    rand_seed = 1,
    quantiles = {'low':0.1, 'high':0.9},
)

# Define the scenarios
interv_day = '2020-04-04'
scenarios = {'baseline': {
              'name':'Baseline',
              'pars': {
                  'interventions': None,
                  }
              },
            'distance': {
              'name':'Social distancing',
              'pars': {
                  'interventions': cv.change_beta(days=interv_day, changes=0.7)
                  }
              },
            'ttq': {
              'name':'Test-trace-quarantine',
              'pars': {
                  'interventions': [
                        cv.test_prob(start_day=interv_day, symp_prob=0.2, asymp_prob=0.05, test_delay=1.0),
                        cv.contact_tracing(start_day=interv_day, trace_probs=0.8, trace_time=1.0),
                    ]
                  }
              },
             }

if __name__ == "__main__": # Required for parallel processing on Windows

    scens = cv.Scenarios(basepars=basepars, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose)
    if do_plot:
        fig1 = scens.plot(do_show=do_show)

