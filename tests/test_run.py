'''
Simple example usage for the Covid-19 agent-based model
'''

#%% Imports and settings
import os
import sciris as sc
import covasim as cv

do_plot = 1
do_save = 0
do_show = 1
debug   = 1
verbose = 0

#%% Define the tests


def test_singlerun():
    sc.heading('Single run test')

    iterpars = {'beta': 0.035,
                }

    sim = cv.Sim(verbose=verbose)
    sim['n_days'] = 20
    sim['pop_size'] = 1000
    sim = cv.single_run(sim=sim, **iterpars)

    return sim


def test_multirun(do_plot=False): # If being run via pytest, turn off
    sc.heading('Multirun test')

    n_days = 60
    pop_size= 1000

    # Method 1 -- Note: this runs 3 simulations, not 3x3!
    iterpars = {'beta': [0.015, 0.025, 0.035],
                'iso_factor': [0.1, 0.5, 0.9],
                }
    sim = cv.Sim(n_days=n_days, pop_size=pop_size)
    sims = cv.multi_run(sim=sim, iterpars=iterpars, verbose=verbose)

    # Method 2 -- run a list of sims
    simlist = []
    for i in range(len(iterpars['beta'])):
        sim = cv.Sim(n_days=n_days, pop_size=pop_size, beta=iterpars['beta'][i], iso_factor=iterpars['iso_factor'][i])
        simlist.append(sim)
    sims2 = cv.multi_run(sim=simlist, verbose=verbose)

    if do_plot:
        for sim in sims + sims2:
            sim.plot()

    return sims


def test_multisim_reduce(do_plot=False): # If being run via pytest, turn off
    sc.heading('Combine results test')

    n_runs = 3
    pop_size = 1000
    pop_infected = 10

    print('Running first sim...')
    sim = cv.Sim(pop_size=pop_size, pop_infected=pop_infected)
    msim = cv.MultiSim(sim, n_runs=n_runs, noise=0.1)
    msim.run(verbose=verbose)
    msim.reduce()
    if do_plot:
        msim.plot()

    return msim


def test_multisim_combine(do_plot=False): # If being run via pytest, turn off
    sc.heading('Combine results test')

    n_runs = 3
    pop_size = 1000
    pop_infected = 10

    print('Running first sim...')
    sim = cv.Sim(pop_size=pop_size, pop_infected=pop_infected, verbose=verbose)
    msim = cv.MultiSim(sim)
    msim.run(n_runs=n_runs, keep_people=True)
    sim1 = msim.combine(output=True)
    assert sim1['pop_size'] == pop_size*n_runs

    print('Running second sim, results should be similar but not identical (stochastic differences)...')
    sim2 = cv.Sim(pop_size=pop_size*n_runs, pop_infected=pop_infected*n_runs)
    sim2.run(verbose=verbose)

    if do_plot:
        msim.plot()
        sim2.plot()

    return msim


def test_simple_scenarios(do_plot=False):
    sc.heading('Simple scenarios test')
    basepars = {'pop_size':1000}

    json_path = 'scen_test.json'
    xlsx_path = 'scen_test.xlsx'

    scens = cv.Scenarios(basepars=basepars)
    scens.run(verbose=verbose)
    if do_plot:
        scens.plot()
    scens.to_json(json_path)
    scens.to_excel(xlsx_path)

    for path in [json_path, xlsx_path]:
        print(f'Removing {path}')
        os.remove(path)
    return scens


def test_complex_scenarios(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test impact of reducing delay time for finding contacts of positives')

    n_runs = 3
    base_pars = {
      'pop_size': 1000,
      'pop_type': 'hybrid',
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
            'name': 'Poor contact tracing',
            'pars': {
                'quar_factor': {'h': 1, 's': 0.5, 'w': 0.5, 'c': 0.25},
                'quar_period': 7,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 0, 's': 0, 'w': 0, 'c': 0},
                        trace_time  = {'h': 1, 's': 7,   'w': 7,   'c': 7})]
            }
        },
        'modtrace': {
            'name': 'Moderate contact tracing',
            'pars': {
                'quar_factor': {'h': 0.75, 's': 0.25, 'w': 0.25, 'c': 0.1},
                'quar_period': 10,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 1, 's': 0.8, 'w': 0.5, 'c': 0.1},
                        trace_time  = {'h': 0,  's': 3,  'w': 3,   'c': 8})]
            }
        },
        'hightrace': {
            'name': 'Fast contact tracing',
            'pars': {
                'quar_factor': {'h': 0.5, 's': 0.1, 'w': 0.1, 'c': 0.1},
                'quar_period': 14,
                'interventions': [cv.test_num(daily_tests=daily_tests),
                cv.contact_tracing(trace_probs = {'h': 1, 's': 0.8, 'w': 0.8, 'c': 0.2},
                        trace_time  = {'h': 0, 's': 1,   'w': 1,   'c': 5})]
            }
        },
        'alltrace': {
            'name': 'Same-day contact tracing',
            'pars': {
                'quar_factor': {'h': 0.0, 's': 0.0, 'w': 0.0, 'c': 0.0},
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
            'cum_severe',
            'n_quarantined',
            'new_quarantined'
        ]
        fig_args = dict(figsize=(24,16))
        scens.plot(do_save=do_save, do_show=do_show, to_plot=to_plot, fig_path=fig_path, n_cols=2, fig_args=fig_args)

    return scens


#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()

    sim1   = test_singlerun()
    sims2  = test_multirun(do_plot=do_plot)
    msim1  = test_multisim_reduce(do_plot=do_plot)
    msim2  = test_multisim_combine(do_plot=do_plot)
    scens1 = test_simple_scenarios(do_plot=do_plot)
    scens2 = test_complex_scenarios(do_plot=do_plot)

    sc.toc(T)
    print('Done.')
