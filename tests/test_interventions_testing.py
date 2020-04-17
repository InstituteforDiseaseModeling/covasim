'''
Testing the effect of testing interventions in Covasim
'''

#%% Imports and settings
import sciris as sc
import covasim as cv

do_plot   = 1
do_show   = 1
do_save   = 0
debug     = 1
keep_sims = 0
fig_paths = [f'results/testing_scen_{i}.png' for i in range(3)]


def test_interventions(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test of testing interventions')


    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1
    base_pars = {
      'pop_size': 1000,
      'use_layers': True,
      }

    base_sim = cv.Sim(base_pars) # create sim object
    n_people = base_sim['pop_size']
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
                                cv.dynamic_pars({'quar_eff':{'days':20, 'vals':[{'h':0.1, 's':0.1, 'w':0.1, 'c':0.1}]}})] # This means that people who've been in contact with known positives isolate with 90% effectiveness
              }
          },
        'floating': {
            'name': 'Test with constant probability based on symptoms',
            'pars': {
                'interventions': cv.test_prob(symp_prob=max_optimistic_testing, asymp_prob=0.0)
                }
        },
        # 'historical': {
        #     'name': 'Test a known number of positive cases',
        #     'pars': {
        #         'interventions': cv.test_historical(n_tests=[100]*npts, n_positive = [1]*npts)
        #     }
        # },
        'sequence': {
            'name': 'Historical switching to probability',
            'pars': {
                'interventions': cv.sequence(days=[10, 51], interventions=[
                    cv.test_num(daily_tests=[1000]*npts),
                    cv.test_prob(symp_prob=0.2, asymp_prob=0.002),
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
      'pop_size': 5000,
      'use_layers': True,
      }

    base_sim = cv.Sim(base_pars) # create sim object
    n_people = base_sim['pop_size']
    npts = base_sim.npts

    # Define overall testing assumptions
    testing_prop = 0.1 # Assumes we could test 10% of the population daily (!!)
    daily_tests = [testing_prop*n_people]*npts # Number of daily tests

    # Define the scenarios
    scenarios = {
        f'{d}dayturnaround': {
            'name':f'Symptomatic testing with {d} days to get results',
            'pars': {
                'interventions': cv.test_num(daily_tests=daily_tests, test_delay=d)
            }
        } for d in range(1, 3+1, 2)
    }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    to_plot = ['cum_infections', 'n_infectious', 'new_tests', 'new_diagnoses']
    fig_args = dict(figsize=(20, 24))

    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path=fig_path, interval=7, fig_args=fig_args, to_plot=to_plot)

    return scens


def test_tracedelay(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test impact of reducing delay time for finding contacts of positives')

    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1
    base_pars = {
      'pop_size': 5000,
      'use_layers': True,
      }

    base_sim = cv.Sim(base_pars) # create sim object
    base_sim['n_days'] = 50
    base_sim['beta'] = 0.03 # Increase beta

    n_people = base_sim['pop_size']
    npts = base_sim.npts


    # Define overall testing assumptions
    testing_prop = 0.1 # Assumes we could test 10% of the population daily (way too optimistic!!)
    daily_tests = [testing_prop*n_people]*npts # Number of daily tests

    # Define the scenarios
    scenarios = {
        'lowtrace': {
            'name': 'Poor contact tracing; 7d quarantine; 50% acquision reduction',
            'pars': {
                'quar_eff': {'h': 1, 's': 0.5, 'w': 0.5, 'c': 0.25},
                'quar_period': 7,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 0, 's': 0, 'w': 0, 'c': 0},
                        trace_time  = {'h': 1, 's': 7,   'w': 7,   'c': 7})]
            }
        },
        'modtrace': {
            'name': 'Moderate contact tracing; 10d quarantine; 75% acquision reduction',
            'pars': {
                'quar_eff': {'h': 0.75, 's': 0.25, 'w': 0.25, 'c': 0.1},
                'quar_period': 10,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 1, 's': 0.8, 'w': 0.5, 'c': 0.1},
                        trace_time  = {'h': 0,  's': 3,  'w': 3,   'c': 8})]
            }
        },
        'hightrace': {
            'name': 'Fast contact tracing; 14d quarantine; 90% acquision reduction',
            'pars': {
                'quar_eff': {'h': 0.5, 's': 0.1, 'w': 0.1, 'c': 0.1},
                'quar_period': 14,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 1, 's': 0.8, 'w': 0.8, 'c': 0.2},
                        trace_time  = {'h': 0, 's': 1,   'w': 1,   'c': 5})]
            }
        },
        'alltrace': {
            'name': 'Same-day contact tracing; 21d quarantine; 100% acquision reduction',
            'pars': {
                'quar_eff': {'h': 0.0, 's': 0.0, 'w': 0.0, 'c': 0.0},
                'quar_period': 21,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 1, 's': 1, 'w': 1, 'c': 1},
                        trace_time  = {'h': 0, 's': 1, 'w': 1, 'c': 2})]
            }
        },
    }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        to_plot = [
            'cum_infections',
            'cum_recoveries',
            'new_infections',
            'n_quarantined',
            'new_quarantined'
        ]
        fig_args = dict(figsize=(24,16))
        scens.plot(do_save=do_save, do_show=do_show, to_plot=to_plot, fig_path=fig_path, n_cols=2, fig_args=fig_args)

    return scens



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    scens1 = test_interventions(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_paths[0])
    scens2 = test_turnaround(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_paths[1])
    scens3 = test_tracedelay(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_paths[2])

    sc.toc()


print('Done.')
