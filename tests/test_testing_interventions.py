'''
Testing the effect of testing interventions in Covasim
'''

#%% Imports and settings
import sciris as sc
import covasim as cv

doplot = 1

def test_interventions(doplot=False):
    sc.heading('Minimal sim test')


    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1

    base_sim = cv.Sim() # create sim object
    n_people = base_sim['n']
    npts = base_sim.npts

    # Define the scenarios
    scenarios = {
        'baseline': {
          'name':'Status quo, no testing',
          'pars': {
              'interventions': None,
              }
          },
        'test1pc': {
          'name':'Test 1% (untargeted); isolate positives',
          'pars': {
              'interventions': cv.TestNum(npts, daily_tests=[0.01*n_people]*npts),
              }
          },
        'test10pc': {
          'name':'Test 10% (untargeted); isolate positives',
          'pars': {
              'interventions': cv.TestNum(npts, daily_tests=[0.10*n_people]*npts),
              }
          },
        'tracing1pc': {
          'name':'Test 1% (contact tracing); isolate positives',
          'pars': {
              'interventions': cv.TestNum(npts, daily_tests=[0.01*n_people]*npts),
              'cont_factor': 0.1, # This means that people who've been in contact with known positives isolate with 90% effectiveness
              }
          },
        'tracing10pc': {
          'name':'TTest 10% (contact tracing); isolate positives',
          'pars': {
              'interventions': cv.TestNum(npts, daily_tests=[0.10*n_people]*npts),
              'cont_factor': 0.1, # This means that people who've been in contact with known positives isolate with 90% effectiveness
              }
          },
        'floating': {
          'name':'Test a constant proportion of the population',
          'pars': {
              'interventions': cv.TestProp(npts, symptomatic_prob=0.9, asymptomatic_prob=0.0, trace_prob=0.9)
              }
          },
         }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose)

    return scens


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    scens = test_interventions(doplot=doplot)

    sc.toc()


print('Done.')