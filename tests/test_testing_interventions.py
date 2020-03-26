'''
Testing the effect of testing interventions in Covasim
'''

#%% Imports and settings
import sciris as sc
import covasim as cova

doplot = 0

def test_interventions(doplot=False):
    sc.heading('Minimal sim test')


    sc.heading('Setting up...')

    sc.tic()

    # Specify what to run!
    scenarios = {
        'baseline':     'Status quo, no testing',
        'test1pc':      'Test 1% (untargeted); isolate positives',
        'test10pc':     'Test 10% (untargeted); isolate positives',
        'tracing1pc':   'Test 1% (contact tracing); isolate positives',
        'tracing10pc':  'Test 10% (contact tracing); isolate positives',
        'floating':     'Known probability of testing',
    }

    # Define the scenarios
    scenarios = {'baseline': {
                  'name':'Baseline',
                  'pars': {
                      'interv_days': [],
                      'interv_effs': [],
                      }
                  },
                'test1pc': {
                  'name':'Test 1% of the population',
                  'pars': {
                      'interventions'] = [cova.FixedTestIntervention(scen_sim, daily_tests=[0.01*n_people]*scen_sim.npts)]
                      }
                  },
                 }


    scens = cova.Scenarios(metapars=metapars, scenarios=scenarios)
    scens.run(keep_sims=keep_sims, verbose=verbose)


    for scenkey,scenname in scenarios.items():

        scen_sim = cova.Sim() # create sim object
        scen_sim.set_seed(seed)
        n_people = scen_sim['n']

        if scenkey == 'baseline':
            scen_sim['interventions'] = []

        elif scenkey == 'test1pc':
            scen_sim['interventions'] = [cova.FixedTestIntervention(scen_sim, daily_tests=[0.01*n_people]*scen_sim.npts)]

        elif scenkey == 'test10pc':
            scen_sim['interventions'] = [cova.FixedTestIntervention(scen_sim, daily_tests=[0.1*n_people]*scen_sim.npts)]

        elif scenkey == 'tracing1pc':
            scen_sim['interventions'] = [cova.FixedTestIntervention(scen_sim, daily_tests=[0.01*n_people]*scen_sim.npts, trace_test=100)]
            scen_sim['cont_factor'] = 0.1 # This means that people who've been in contact with known positives isolate with 90% effectiveness

        elif scenkey == 'tracing10pc':
            scen_sim['interventions'] = [cova.FixedTestIntervention(scen_sim, daily_tests=[0.1*n_people]*scen_sim.npts, trace_test=100)]
            scen_sim['cont_factor'] = 0.1 # This means that people who've been in contact with known positives isolate with 90% effectiveness

        elif scenkey == 'floating':
            scen_sim['interventions'] = [cova.FloatingTestIntervention(scen_sim, symptomatic_probability=0.9, asymptomatic_probability=0.00, trace_probability=0.9)]


        scen_sim.run(verbose=verbose)

        for reskey in reskeys:
            allres[reskey][scenkey]['name'] = scenname
            allres[reskey][scenkey]['values'] = scen_sim.results[reskey].values


    #%% Print statistics
    for reskey in reskeys:
        for scenkey in list(scenarios.keys()):
            print(f'{reskey} {scenkey}: {allres[reskey][scenkey]["values"][-1]:0.0f}')

    return scens


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    scens = test_interventions(doplot=doplot)

    sc.toc()


print('Done.')