'''
Testing the effect of testing interventions in Covasim
'''

#%% Imports and settings
import sciris as sc
import covasim as cv

do_plot   = 1
do_show   = 0
do_save   = 0
debug     = 1
keep_sims = 0
fig_path  = [f'results/testing_scen_{i}.png' for i in range(3)]

def test_interventions(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test of testing interventions')


    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1
    base_pars = {
      'n': 1000
      }

    base_sim = cv.Sim(base_pars) # create sim object
    n_people = base_sim['n']
    npts = base_sim.npts

    # Define overall testing assumptions
    # As the most optimistic case, we assume countries could get to South Korea's testing levels. S Korea has tested
    # an average of 10000 people/day over March, or 270,000 in total. This is ~200 people per million every day (0.02%).
    max_optimistic_testing = 0.0002
    optimistic_daily_tests = [max_optimistic_testing*n_people]*npts # Very best-case scenario for asymptomatic testing

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
              'interventions': cv.test_num(daily_tests=optimistic_daily_tests)
              }
          },
        'tracing': {
          'name':'Assuming South Korea testing levels of 0.02% daily (with contact tracing); isolate positives',
          'pars': {
              'interventions': [cv.test_num(daily_tests=optimistic_daily_tests),
                                cv.dynamic_pars({'cont_factor':{'days':20, 'vals':0.1}})] # This means that people who've been in contact with known positives isolate with 90% effectiveness
              }
          },
        'floating': {
            'name': 'Test with constant probability based on symptoms',
            'pars': {
                'interventions': cv.test_prob(symptomatic_prob=max_optimistic_testing, asymptomatic_prob=0.0, trace_prob=0.9)
                }
        },
        'historical': {
            'name': 'Test a known number of positive cases',
            'pars': {
                'interventions': cv.test_historical(n_tests=[100]*npts, n_positive = [1]*npts)
            }
        },
        'sequence': {
            'name': 'Historical switching to probability',
            'pars': {
                'interventions': cv.sequence(days=[10, 51], interventions=[
                    cv.test_historical(n_tests=[100] * npts, n_positive=[1] * npts),
                    cv.test_prob(symptomatic_prob=0.2, asymptomatic_prob=0.002, trace_prob=0.9),
                ])
            }
        },

    }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path=fig_path)

    return scens


def test_turnaround(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test impact of reducing delay time for getting test results')

    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1
    base_pars = {
      'n': 20000
      }

    base_sim = cv.Sim(base_pars) # create sim object
    n_people = base_sim['n']
    npts = base_sim.npts

    # Define overall testing assumptions
    testing_prop = 0.1 # Assumes we could test 10% of the population daily (!!)
    daily_tests = [testing_prop*n_people]*npts # Number of daily tests

    # Define the scenarios
    scenarios = {
        '7dayturnaround': {
          'name':'Symptomatic testing with 7 days to get results',
            'pars': {
                'interventions': cv.test_num(daily_tests=daily_tests, test_delay=7)
              }
          },
        '3dayturnaround': {
          'name':'Symptomatic testing with 3 days to get results',
          'pars': {
              'interventions': cv.test_num(daily_tests=daily_tests, test_delay=3)
              }
          },
        '0dayturnaround': {
            'name': 'Symptomatic testing with immediate results',
            'pars': {
                'interventions': cv.test_num(daily_tests=daily_tests, test_delay=0)
            }
        },
    }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path=fig_path)

    return scens



def test_tracedelay(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test impact of reducing delay time for finding contacts of positives')

    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1
    base_pars = {
      'n': 20000
      }

    base_sim = cv.Sim(base_pars) # create sim object
    n_people = base_sim['n']
    npts = base_sim.npts

    # Define overall testing assumptions
    testing_prop = 0.1 # Assumes we could test 10% of the population daily (way too optimistic!!)
    daily_tests = [testing_prop*n_people]*npts # Number of daily tests

    # Define the scenarios
    scenarios = {
        '7daytrace': {
          'name': 'Test 10% of population every day; 7 days to find their contacts, who then isolate with 50% effectiveness',
            'pars': {
                'cont_time': 7,
                'cont_factor': 0.5,
                'interventions': cv.test_num(daily_tests=daily_tests)
              }
          },
        '3daytrace': {
          'name': 'Test 10% of population every day; 3 days to find their contacts, who then isolate with 75% effectiveness',
          'pars': {
              'cont_time': 3,
              'cont_factor': 0.25,
              'interventions': cv.test_num(daily_tests=daily_tests)
              }
          },
        '1daytrace': {
            'name': 'Test 10% of population every day; 1 day to find their contacts, who then isolate with 90% effectiveness',
            'pars': {
                'cont_time': 1,
                'cont_factor': 0.1,
                'interventions': cv.test_num(daily_tests=daily_tests)
            }
        },
    }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path=fig_path)

    return scens



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    scens1 = test_interventions(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_path[0])
    scens2 = test_turnaround(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_path[1])
    scens3 = test_tracedelay(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_path[2])

    sc.toc()


print('Done.')
