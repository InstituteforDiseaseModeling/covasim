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

#%% Define the tests

def test_singlerun():
    sc.heading('Single run test')

    iterpars = {'beta': 0.035,
                }

    sim = cv.Sim()
    sim['n_days'] = 20
    sim['n'] = 1000
    sim = cv.single_run(sim=sim, **iterpars)

    return sim


def test_combine(do_plot=False): # If being run via pytest, turn off
    sc.heading('Combine results test')

    n_runs = 3
    n = 1000
    n_infected = 10

    print('Running first sim...')
    sim = cv.Sim({'n':n, 'n_infected':n_infected})
    sim = cv.multi_run(sim=sim, n_runs=n_runs, combine=True)
    assert len(sim.people) == n*n_runs

    print('Running second sim, results should be similar but not identical (stochastic differences)...')
    sim2 = cv.Sim({'n':n*n_runs, 'n_infected':n_infected*n_runs})
    sim2.run()

    if do_plot:
        sim.plot()
        sim2.plot()

    return sim


def test_multirun(do_plot=False): # If being run via pytest, turn off
    sc.heading('Multirun test')

    # Note: this runs 3 simulations, not 3x3!
    iterpars = {'beta': [0.015, 0.025, 0.035],
                'cont_factor': [0.1, 0.5, 0.9],
                }

    sim = cv.Sim()
    sim['n_days'] = 60
    sim['n'] = 1000
    sims = cv.multi_run(sim=sim, iterpars=iterpars)

    if do_plot:
        for sim in sims:
            sim.plot()

    return sims


def test_scenarios(do_plot=False):
    sc.heading('Scenarios test')
    basepars = {'n':1000}

    json_path = 'scen_test.json'
    xlsx_path = 'scen_test.xlsx'

    scens = cv.Scenarios(basepars=basepars)
    scens.run()
    if do_plot:
        scens.plot()
    scens.to_json(json_path)
    scens.to_excel(xlsx_path)

    for path in [json_path, xlsx_path]:
        print(f'Removing {path}')
        os.remove(path)
    return scens



#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()

    # sim2  = test_singlerun()
    # sim3  = test_combine(do_plot=do_plot)
    # sims  = test_multirun(do_plot=do_plot)
    scens = test_scenarios(do_plot=do_plot)

    sc.toc(T)


print('Done.')
