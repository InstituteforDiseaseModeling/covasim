'''
Test resuming a simulation partway, as well as reproducing two simulations with
different initialization states and after saving to disk.
'''

#%% Imports and settings
import os
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

    s0 = cv.Sim(pars, start_day='2020-01-01')
    s1 = s0.copy()
    s0.run()

    # Cannot run the same simulation multiple times
    with pytest.raises(cv.AlreadyRunError):
        s0.run()

    # If until=0 then no timesteps will be taken
    with pytest.raises(cv.AlreadyRunError):
        s1.run(until=0, reset_seed=False)
    assert s1.initialized # It should still have been initialized though

    s1.run(until='2020-01-31', reset_seed=False)
    with pytest.raises(cv.AlreadyRunError):
        s1.run(until=30, reset_seed=False) # Error if running up to the same value
    with pytest.raises(cv.AlreadyRunError):
        s1.run(until=20, reset_seed=False) # Error if running until a previous timestep

    s1.run(until=45, reset_seed=False)
    s1.run(reset_seed=False)

    assert np.all(s0.results['cum_infections'].values == s1.results['cum_infections']) # Results should be identical

    return s1


def test_reset_seed():
    sc.heading('Test that resetting a simulation works')

    until = 30
    s0 = cv.Sim(pars)
    s1 = s0.copy()
    s0.run()

    s1.run(until=until)
    s1.run(reset_seed=True)

    assert     np.all(s0.results['cum_infections'][:until] == s1.results['cum_infections'][:until]) # Results for the first 30 days should be the same
    assert not np.all(s0.results['cum_infections'][until:] == s1.results['cum_infections'][until:]) # Results should be different

    return s1


def test_reproducibility():
    sc.heading('Test that sims are reproducible')

    fn = 'save-load-test.sim' # Name of the test file to save

    #The results of the sim shouldn't be affected by what you do or don't do prior to sim.run()
    s1 = cv.Sim(pars)
    s1.initialize()
    s2 = s1.copy()
    s1.run()
    s2.run()
    r1ci = s1.summary['cum_infections']
    r2ci = s2.summary['cum_infections']
    assert r1ci == r2ci

    # If you run a sim and save it, you should be able to re-run it on load
    s3 = cv.Sim(pars)
    s3.run()
    s3.save(fn)
    s4 = cv.load(fn)
    s4.initialize()
    s4.run()
    r3ci = s3.summary['cum_infections']
    r4ci = s4.summary['cum_infections']
    assert r3ci == r4ci
    if os.path.exists(fn): # Tidy up -- after the assert to allow inspection if it fails
        os.remove(fn)

    return s4


def test_step(): # If being run via pytest, turn off
    sc.heading('Test starting and stopping')

    # Create and run a basic simulation
    s1 = cv.Sim(pars)
    s1.run(verbose=0)

    # Test that step works
    s2 = cv.Sim(pars)
    s2.initialize()
    for n in range(s2.npts):
        s2.step()
    s2.finalize()

    # Compare results
    key = 'cum_infections'
    assert (s1.results[key][:] == s2.results[key][:]).all(), 'Next values do not match'

    return s2


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim1 = test_resuming()
    sim2 = test_reset_seed()
    sim3 = test_reproducibility()
    sim4 = test_step()

    print('\n'*2)
    sc.toc(T)
    print('Done.')