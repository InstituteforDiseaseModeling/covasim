'''
Testing the effect of testing interventions in Covasim
'''

#%% Imports and settings
import matplotlib
matplotlib.use('TkAgg')
import sciris as sc
import covasim as cv

do_plot = 1
do_show = 0
do_save = 1
fig_path = 'results/testing_scens.png'

def test_interventions(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test of testing interventions')


    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1

    base_sim = cv.Sim() # create sim object
    n_people = base_sim['n']
    npts = base_sim.npts

    # Define overall testing assumptions
    # As the most optimistic case, we assume countries could get to South Korea's testing levels. S Korea has tested
    # an average of 10000 people/day over March, or 270,000 in total. This is ~200 people per million every day (0.02%).
    max_optimistic_testing = 0.0002
    optimistic_daily_tests = [max_optimistic_testing*n_people]*npts # Very best-case scenario

    # Define the scenarios
    scenarios = {
        'baseline': {
          'name':'Status quo, no testing',
          'pars': {
              'interventions': None,
              }
          },
        'test_skorea': {
          'name':'Assuming South Korea testing levels of 0.02% daily (untargeted); isolate positives',
          'pars': {
              'interventions': cv.TestNum(npts, daily_tests=optimistic_daily_tests)
              }
          },
        'tracing1pc': {
          'name':'Assuming South Korea testing levels of 0.02% daily (with contact tracing); isolate positives',
          'pars': {
              'interventions': cv.TestNum(npts, daily_tests=optimistic_daily_tests),
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

    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path=fig_path)

    return scens


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    scens = test_interventions(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_path)

    sc.toc()


print('Done.')