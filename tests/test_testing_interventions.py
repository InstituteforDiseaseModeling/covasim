'''
Testing the effect of testing interventions in Covasim
'''

#%% Imports and settings
import matplotlib
matplotlib.use('Agg')
import sciris as sc
import covasim as cv

do_plot   = 1
do_show   = 0
do_save   = 1
debug     = 0
keep_sims = 0
fig_path  = [f'results/testing_scen_{i}.png' for i in range(3)]

def test_interventions(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test of testing interventions')
    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1
    base_pars = {
      'n': 10000
    }

    base_sim = cv.Sim(base_pars) # create sim object
    base_sim['n_seed'] = 1
    base_sim['beta']   = 0.012
    n_people = base_sim['n']
    npts = base_sim.npts

    # Define overall testing assumptions
    testing = 0.005
    daily_tests = [testing*n_people]*npts # Best-case scenario for asymptomatic testing

    # Define the scenarios
    scenarios = {
        'baseline': {
            'name':'Status quo, no testing',
            'pars': {
                'interventions': None,
            }
        },
        'untargeted': {
            'name': f'Assuming {100*testing:.2f}% daily (untargeted); isolate positives 90%',
            'pars': {
                'diag_factor': 0.1,
                'interventions': cv.test_num(daily_tests=daily_tests, sympt_test=1)
            }
        },
        'tracing': {
            'name': f'Assuming {100*testing:.2f}% daily (untargeted); isolate positives 90%; tracing',
            'pars': {
                'diag_factor': 0.1,
                # Contact tracing: 100% at home with 0d dela, 80% of school with 3d delay, 50% of work with 3d delay, 10% community with 3d delay
                'interventions': [
                    cv.test_num(daily_tests=daily_tests, sympt_test=1),
                    cv.contact_tracing(trace_probs = {'h': 1, 's': 0.8, 'w': 0.5, 'c': 0.1},
                                       trace_time  = {'h': 0,  's': 2,  'w': 2,   'c': 3})]
            }
        },
        'floating': {
            'name': f'Test 3% of symptomatics; {100*testing:.2f}% asymptomatics; isolate positives 90%; tracing',
            'pars': {
                'diag_factor': 0.1,
                'interventions': [
                    cv.test_prob(symptomatic_prob=0.03, asymptomatic_prob=testing),
                    cv.contact_tracing(trace_probs = {'h': 1, 's': 0.8, 'w': 0.5, 'c': 0.1},
                                       trace_time  = {'h': 0,  's': 2,  'w': 2,   'c': 3})]
            }
        },
        'sequence': {
            'name': f'Historical switching to probability',
            'pars': {
                'diag_factor': 0.1,
                'interventions': cv.sequence(days=[50, base_sim['n_days']], # Switch at day 20
                    interventions=[
                        cv.test_historical(n_tests=[1000] * npts, n_positive=[100] * npts),
                        cv.test_prob(symptomatic_prob=0.03, asymptomatic_prob=testing),
                    ])
            }
        },

    }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        to_plot = default_scen_plots = [
            'cum_infections',
            'n_infectious',
            'n_quarantined',
        ]
        scens.plot(do_save=do_save, do_show=do_show, to_plot=to_plot, fig_path=fig_path)

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
        f'{d}dayturnaround': {
            'name':f'Symptomatic testing with {d} days to get results',
            'pars': {
                'interventions': cv.test_num(daily_tests=daily_tests, test_delay=d)
            }
        } for d in range(1, 7+1)
    }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        to_plot = default_scen_plots = [
            'cum_infections',
            'n_infectious',
            'cum_deaths',
        ]
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
    base_sim['n_days'] = 50
    n_people = base_sim['n']
    npts = base_sim.npts

    # Define overall testing assumptions
    testing_prop = 0.1 # Assumes we could test 10% of the population daily (way too optimistic!!)
    daily_tests = [testing_prop*n_people]*npts # Number of daily tests

    # Define the scenarios
    scenarios = {
        'lowtrace': {
            'name': '10% daily testing; poor contact tracing; quarantine reduces acq 50%; 7d quarantine',
            'pars': {
                'quar_trans_factor': {'h': 1, 's': 0.5, 'w': 0.5, 'c': 0.25},
                'quar_acq_factor': 0.5,
                'quar_period': 7,
                'interventions': [
                    cv.test_num(daily_tests=daily_tests),
                    cv.contact_tracing(trace_probs = {'h': 1, 's': 0.8, 'w': 0.5, 'c': 0.0},
                                       trace_time  = {'h': 0, 's': 7,   'w': 7,   'c': 0})]
            }
        },
        'modtrace': {
            'name': '10% daily testing; moderate contact tracing; quarantine reduces acq 75%; 10d quarantine',
            'pars': {
                'quar_trans_factor': {'h': 1, 's': 0.25, 'w': 0.25, 'c': 0.1},
                'quar_acq_factor': 0.75,
                'quar_period': 10,
                'interventions': [
                    cv.test_num(daily_tests=daily_tests),
                    cv.contact_tracing(trace_probs = {'h': 1, 's': 0.8, 'w': 0.5, 'c': 0.1},
                                       trace_time  = {'h': 0,  's': 3,  'w': 3,   'c': 8})]
            }
        },
        'hightrace': {
            'name': '10% daily testing; good contact tracing; quarantine reduces acq 90%; 14d quarantine',
            'pars': {
                'quar_trans_factor': {'h': 0.5, 's': 0.1, 'w': 0.1, 'c': 0.1},
                'quar_acq_factor': 0.9,
                'quar_period': 14,
                'interventions': [
                    cv.test_num(daily_tests=daily_tests),
                    cv.contact_tracing(trace_probs = {'h': 1, 's': 0.8, 'w': 0.8, 'c': 0.2},
                                       trace_time  = {'h': 0, 's': 1,   'w': 1,   'c': 5})]
            }
        },
        'crazy': {
            'name': '10% daily testing; perfect same-day contact tracing; quarantine reduces stops acq; 21d quarantine',
            'pars': {
                'quar_trans_factor': {'h': 0.0, 's': 0.0, 'w': 0.0, 'c': 0.0},
                'quar_acq_factor': 1,
                'quar_period': 21,
                'interventions': [
                    cv.test_num(daily_tests=daily_tests),
                    cv.contact_tracing(trace_probs = {'h': 1, 's': 1, 'w': 1, 'c': 1},
                                       trace_time  = {'h': 0, 's': 0, 'w': 0, 'c': 0})]
            }
        },
    }

    metapars = {'n_runs': n_runs}

    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        to_plot = default_scen_plots = [
            'cum_infections',
            'n_diagnosed',
            'n_quarantined',
        ]
        scens.plot(do_save=do_save, do_show=do_show, to_plot=to_plot, fig_path=fig_path)

    return scens




#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    #scens1 = test_interventions(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_path[0])
    scens2 = test_turnaround(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_path[1])
    scens3 = test_tracedelay(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_path[2])

    sc.toc()


print('Done.')
