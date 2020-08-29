'''
Test partial simulation
'''

#%% Imports and settings
import pytest
import numpy as np
import numba as nb
import pylab as pl
import sciris as sc
import covasim as cv


doplot = 0


#%% Define the tests

def test_resuming():
    s0 = cv.Sim()
    s1 = s0.copy()
    s0.run()

    with pytest.raises(cv.TimestepsExhaustedError):
        # Cannot run the same simulation multiple times
        s0.run()

    with pytest.raises(cv.TimestepsExhaustedError):
        # If until=0 then no timesteps will be taken
        s1.run(until=0)
    assert s1.initialized # It should still have been initialized though

    s1.run(until=30)
    with pytest.raises(cv.TimestepsExhaustedError):
        s1.run(until=30) # Error if running up to the same value
    with pytest.raises(cv.TimestepsExhaustedError):
        s1.run(until=20) # Error if running until a previous timestep

    s1.run(until=45)
    s1.run()

    assert np.all(s0.results['cum_infections'].values == s1.results['cum_infections']) # Results should be identical

def test_reset_seed():
    s0 = cv.Sim()
    s1 = s0.copy()
    s0.run()

    s1.run(until=30)
    s1.run(reset_seed=True)

    assert not np.all(s0.results['cum_infections'].values == s1.results['cum_infections']) # Results should be different
    assert np.all(s0.results['cum_infections'].values[0:30] == s1.results['cum_infections'][0:30]) # Results for the first 30 days should be the same
    assert s0.results['cum_infections'].values[31] != s1.results['cum_infections'][31] # Results on day 31 should be different

#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    test_resuming()
    test_reset_seed()

    print('\n'*2)
    sc.toc(T)
    print('Done.')