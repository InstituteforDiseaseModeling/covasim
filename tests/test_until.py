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

    return s1

def test_reset_seed():
    s0 = cv.Sim()
    s1 = s0.copy()
    s0.run()

    s1.run(until=30)
    s1.run(reset_seed=True)

    assert not np.all(s0.results['cum_infections'].values == s1.results['cum_infections']) # Results should be different
    assert np.all(s0.results['cum_infections'].values[0:30] == s1.results['cum_infections'][0:30]) # Results for the first 30 days should be the same
    assert s0.results['cum_infections'].values[31] != s1.results['cum_infections'][31] # Results on day 31 should be different

    return s1

#%% CK tests

pars = dict(pop_size=1e3, verbose=0)


def test_until_date():
    ''' Test that until can be a date '''
    sim = cv.Sim(**pars)
    sim.run(until='2020-04-01')
    sim.run()
    return sim


def test_reproducibility():
    ''' The results of the sim shouldn't be affected by what you do or don't do prior to sim.run() '''
    s1 = cv.Sim(**pars)
    s1.initialize()
    s2 = s1.copy()
    s1.run()
    s2.run()
    r1ci = s1.summary['cum_infections']
    r2ci = s2.summary['cum_infections']
    test = r1ci == r2ci
    if test:
        print('Sim is reproducible')
    else:
        raise Exception(f'Sim is NOT reproducible: {r1ci} vs {r2ci}')
    return s2


def test_run_from_load():
    ''' If you run a sim and save it, you should be able to re-run it on load '''
    fn = 'save-load-test.sim'
    s1 = cv.Sim(**pars)
    s1.run()
    s1.save(fn)
    s2 = cv.load(fn)
    s2.run()
    r1ci = s1.summary['cum_infections']
    r2ci = s2.summary['cum_infections']
    test = r1ci == r2ci
    if test:
        print('Sim is reproducible')
    else:
        raise Exception(f'Sim is NOT reproducible: {r1ci} vs {r2ci}')
    return s2



#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sa = test_resuming()
    sb = test_reset_seed()

    sc = test_until_date()
    sd = test_reproducibility()
    se = test_run_from_load()

    print('\n'*2)
    sc.toc(T)
    print('Done.')