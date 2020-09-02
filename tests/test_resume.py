'''
Test partial simulation
'''

#%% Imports and settings
import pytest
import numpy as np
import sciris as sc
import covasim as cv

# Simulation and test parameters
doplot = 0
pars = dict(pop_size=1e3, verbose=0)


#%% Define the tests

def test_resuming():
    sc.heading('Test that resuming a run works')

    s0 = cv.Sim(**pars, start_day='2020-01-01')
    s1 = s0.copy()
    s0.run()

    # Cannot run the same simulation multiple times
    with pytest.raises(cv.AlreadyRunError):
        s0.run()

    # If until=0 then no timesteps will be taken
    with pytest.raises(cv.AlreadyRunError):
        s1.run(until=0)
    assert s1.initialized # It should still have been initialized though

    s1.run(until='2020-01-30', reset_seed=False)
    with pytest.raises(cv.AlreadyRunError):
        s1.run(until=30) # Error if running up to the same value
    with pytest.raises(cv.AlreadyRunError):
        s1.run(until=20) # Error if running until a previous timestep

    s1.run(until=45, reset_seed=False)
    s1.run(reset_seed=False)

    assert np.all(s0.results['cum_infections'].values == s1.results['cum_infections']) # Results should be identical

    return s1


def test_reset_seed():
    sc.heading('Test that resetting a simulation works')
    s0 = cv.Sim(**pars)
    s1 = s0.copy()
    s0.run()

    s1.run(until=30)
    s1.run(reset_seed=True)

    assert not np.all(s0.results['cum_infections'].values == s1.results['cum_infections']) # Results should be different
    assert np.all(s0.results['cum_infections'].values[0:30] == s1.results['cum_infections'][0:30]) # Results for the first 30 days should be the same
    assert s0.results['cum_infections'].values[31] != s1.results['cum_infections'][31] # Results on day 31 should be different

    return s1


def test_reproducibility():
    sc.heading('Test that sims are reproducible')

    #The results of the sim shouldn't be affected by what you do or don't do prior to sim.run()
    s1 = cv.Sim(**pars)
    s1.initialize()
    s2 = s1.copy()
    s1.run()
    s2.run()
    r1ci = s1.summary['cum_infections']
    r2ci = s2.summary['cum_infections']
    assert r1ci == r2ci

    # If you run a sim and save it, you should be able to re-run it on load
    fn = 'save-load-test.sim'
    s1 = cv.Sim(**pars)
    s1.run()
    s1.save(fn)
    s2 = cv.load(fn)
    s2.initialize()
    s2.run()
    r1ci = s1.summary['cum_infections']
    r2ci = s2.summary['cum_infections']
    assert r1ci == r2ci

    return s2


def test_start_stop(): # If being run via pytest, turn off
    sc.heading('Test starting and stopping')

    pars = {'pop_size': 1000}

    # Create and run a basic simulation
    sim1 = cv.Sim(pars)
    sim1.run(verbose=0)

    # Test that step works
    sim2 = cv.Sim(pars)
    sim2.initialize()
    for n in range(sim2.npts):
        sim2.step()
    sim2.finalize()

    # Test that until works
    sim3 = cv.Sim(pars)
    sim3.run(until=20)
    sim3.run(reset_seed=False)

    # Compare results
    key = 'cum_infections'
    assert (sim1.results[key][:] == sim2.results[key][:]).all(), 'Next values do not match'
    assert (sim1.results[key][:] == sim3.results[key][:]).all(), 'Until values do not match'

    return sim2


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim1 = test_resuming()
    sim2 = test_reset_seed()
    sim3 = test_reproducibility()
    sim4 = test_start_stop()

    print('\n'*2)
    sc.toc(T)
    print('Done.')