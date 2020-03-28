'''
Simple example usage for the Covid-19 agent-based model
'''

#%% Imports and settings
import os
import pytest
import sciris as sc
import covasim as cv

doplot = 1


#%% Define the tests

def test_parsobj():
    sc.heading('Testing parameters object')

    pars1 = {'a':1, 'b':2}
    parsobj = cv.ParsObj(pars1)

    # Once created, you cannot directly add new keys to a parsobj, and a nonexistent key works like a dict
    with pytest.raises(KeyError): parsobj['c'] = 3
    with pytest.raises(KeyError): parsobj['c']

    # Only a dict is allowed
    with pytest.raises(TypeError):
        pars2 = ['a', 'b']
        cv.ParsObj(pars2)

    return parsobj


def test_microsim():
    sc.heading('Minimal sim test')

    sim = cv.Sim()
    pars = {
        'n': 10,
        'n_infected': 1,
        'contacts': 2,
        'n_days': 10
        }
    sim.update_pars(pars)
    sim.run()

    return sim


def test_sim(doplot=False): # If being run via pytest, turn off
    sc.heading('Basic sim test')

    # Settings
    seed = 1
    verbose = 1

    # Create and run the simulation
    sim = cv.Sim()
    sim.set_seed(seed)
    sim.run(verbose=verbose)

    # Optionally plot
    if doplot:
        sim.plot()

    return sim


def test_singlerun():
    sc.heading('Single run test')

    iterpars = {'beta': 0.035,
                'incub': 8,
                }

    sim = cv.Sim()
    sim['n_days'] = 20
    sim = cv.single_run(sim=sim, **iterpars)

    return sim


def test_combine(doplot=False): # If being run via pytest, turn off
    sc.heading('Combine results test')

    n_runs = 5
    n = 2000
    n_infected = 100

    print('Running first sim...')
    sim = cv.Sim({'n':n, 'n_infected':n_infected})
    sim = cv.multi_run(sim=sim, n_runs=n_runs, combine=True)
    assert len(sim.people) == n*n_runs

    print('Running second sim, results should be similar but not identical (stochastic differences)...')
    sim2 = cv.Sim({'n':n*n_runs, 'n_infected':n_infected*n_runs})
    sim2.run()

    if doplot:
        sim.plot()
        sim2.plot()

    return sim


def test_multirun(doplot=False): # If being run via pytest, turn off
    sc.heading('Multirun test')

    # Note: this runs 3 simulations, not 3x3!
    iterpars = {'beta': [0.015, 0.025, 0.035],
                'incub': [4, 5, 6],
                }

    sim = cv.Sim() # Shouldn't be necessary, but is for now
    sim['n_days'] = 60
    sims = cv.multi_run(sim=sim, iterpars=iterpars)

    if doplot:
        for sim in sims:
            sim.plot()

    return sims


def test_scenarios(doplot=False):
    sc.heading('Scenarios test')
    scens = cv.Scenarios()
    scens.run()
    if doplot:
        scens.plot()
    return scens


def test_fileio():
    sc.heading('Test file saving')

    json_path = 'test_covasim.json'
    xlsx_path = 'test_covasim.xlsx'

    # Create and run the simulation
    sim = cv.Sim()
    sim['n_days'] = 20
    sim.run(verbose=0)

    # Create objects
    json = sim.to_json()
    xlsx = sim.to_xlsx()
    print(xlsx)

    # Save files
    sim.to_json(json_path)
    sim.to_xlsx(xlsx_path)

    for path in [json_path, xlsx_path]:
        print(f'Removing {path}')
        os.remove(path)

    return json


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    parsobj = test_parsobj()
    sim0    = test_microsim()
    sim1    = test_sim(doplot=doplot)
    sim2    = test_singlerun()
    sim3    = test_combine(doplot=doplot)
    sims    = test_multirun(doplot=doplot)
    scens   = test_scenarios(doplot=doplot)
    json    = test_fileio()

    sc.toc()


print('Done.')
