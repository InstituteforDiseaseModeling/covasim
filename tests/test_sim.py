'''
Simple example usage for the Covid-19 agent-based model
'''

#%% Imports and settings
import os
import pytest
import sciris as sc
import covasim as cv

do_plot = 1
do_save = 0
do_show = 1

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


def test_sim(do_plot=False, do_save=False, do_show=False): # If being run via pytest, turn off
    sc.heading('Basic sim test')

    # Settings
    seed = 1
    verbose = 1

    # Create and run the simulation
    sim = cv.Sim()
    sim.set_seed(seed)
    sim.run(verbose=verbose)

    # Optionally plot
    if do_plot:
        sim.plot(do_save=do_save, do_show=do_show)

    return sim


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
    scens = cv.Scenarios(basepars=basepars)
    scens.run()
    if do_plot:
        scens.plot()
    return scens


def test_fileio():
    sc.heading('Test file saving')

    json_path = 'test_covasim.json'
    xlsx_path = 'test_covasim.xlsx'

    # Create and run the simulation
    sim = cv.Sim()
    sim['n_days'] = 20
    sim['n'] = 1000
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


def test_start_stop(): # If being run via pytest, turn off
    sc.heading('Test starting and stopping')

    pars = {'n': 1000}

    # Create and run a basic simulation
    sim1 = cv.Sim(pars)
    sim1.run(verbose=0)

    # Test start and stop
    stop = 20
    sim2 = cv.Sim(pars)
    sim2.run(start=0, stop=stop, verbose=0)
    sim2.run(start=stop, stop=None, verbose=0)

    # Test that next works
    sim3 = cv.Sim(pars)
    sim3.initialize()
    for n in range(sim3.npts):
        sim3.next(verbose=0)
    sim3.finalize()

    # Compare results
    key = 'cum_infections'
    assert (sim1.results[key][:] == sim2.results[key][:]).all(), 'Start-stop values do not match'
    assert (sim1.results[key][:] == sim3.results[key][:]).all(), 'Next values do not match'

    return sim2


def test_sim_data(do_plot=False, do_show=False):
    sc.heading('Data test')

    pars = dict(
        n=2000,
        start_day = '2020-01-01',
        )

    # Create and run the simulation
    sim = cv.Sim(pars=pars, datafile=os.path.join(sc.thisdir(__file__), 'example_data.csv'))
    sim.run()

    # Optionally plot
    if do_plot:
        sim.plot(do_show=do_show)

    return sim


#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()

    pars  = test_parsobj()
    sim0  = test_microsim()
    sim1  = test_sim(do_plot=do_plot, do_save=do_save, do_show=do_show)
    sim2  = test_singlerun()
    sim3  = test_combine(do_plot=do_plot)
    sims  = test_multirun(do_plot=do_plot)
    scens = test_scenarios(do_plot=do_plot)
    json  = test_fileio()
    sim4  = test_start_stop()
    sim5  = test_sim_data(do_plot=do_plot, do_show=do_show)

    sc.toc(T)


print('Done.')
