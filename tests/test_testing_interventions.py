'''
Testing the effect of testing interventions in Covasim
'''

#%% Imports and settings
import matplotlib
matplotlib.use('TkAgg')
import os
import pytest
import sciris as sc
import covasim as cova
import pylab as pl

doplot = 0

sc.heading('Setting up...')

sc.tic()

# Specify what to run!
do_run = 1
scenarios = {
    'baseline':     'Status quo, no testing',
    'test1pc':      'Test 1% (untargeted); isolate positives',
    'test10pc':     'Test 10% (untargeted); isolate positives',
    'tracing1pc':   'Test 1% (contact tracing); isolate positives',
    'tracing10pc':  'Test 10% (contact tracing); isolate positives',
}

# Other options
do_save = 0 # refers to whether to save plot - see also save_sims
do_plot = 0
show_plot = 0
save_sims = 0 # WARNING, huge! (>100 MB)
verbose = 1
seed = 1
reskeys = ['cum_exposed', 'n_exposed']

version  = 'v0'
date     = '2020mar18'
folder   = 'results'
basename = f'{folder}/covasim_scenarios_{date}_{version}'
fig_path   = f'{basename}.png'


if do_run:

    # Create result storage
    allres = sc.objdict()
    for reskey in reskeys:
        allres[reskey] = sc.objdict()
        for scenkey in scenarios.keys():
            allres[reskey][scenkey] = sc.objdict()
            for nv in ['name', 'value']:
                allres[reskey][scenkey][nv] = None # This will get populated below

    for scenkey,scenname in scenarios.items():

        scen_sim = cova.Sim() # create sim object
        scen_sim.set_seed(seed)
        n_people = scen_sim['n']
        n_days = scen_sim['n_days']

        if scenkey == 'baseline':
            scen_sim['daily_tests'] = [] # No tests

        elif scenkey == 'test1pc':
            scen_sim['daily_tests'] = [0.01*n_people]*n_days

        elif scenkey == 'test10pc':
            scen_sim['daily_tests'] = [0.1*n_people]*n_days

        elif scenkey == 'tracing1pc':
            scen_sim['daily_tests'] = [0.01*n_people]*n_days
            scen_sim['cont_factor'] = 0.1 # This means that people who've been in contact with known positives isolate with 90% effectiveness
            scen_sim['trace_test'] = 100 # This means that people who've been in contact with known positives are 100x more likely to test

        elif scenkey == 'tracing10pc':
            scen_sim['daily_tests'] = [0.1*n_people]*n_days
            scen_sim['cont_factor'] = 0.1 # This means that people who've been in contact with known positives isolate with 90% effectiveness
            scen_sim['trace_test'] = 100 # This means that people who've been in contact with known positives are 100x more likely to test

        scen_sim.run(verbose=verbose)

        for reskey in reskeys:
            allres[reskey][scenkey]['name'] = scenname
            allres[reskey][scenkey]['values'] = scen_sim.results[reskey].values



#%% Print statistics
for reskey in reskeys:
    for scenkey in list(scenarios.keys()):
        print(f'{reskey} {scenkey}: {allres[reskey][scenkey]["values"][-1]:0.0f}')

sc.toc()