'''
Tests of the utilies for the model.
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

def test_rand():

    default_seed = 1

    @nb.njit
    def numba_rand():
        return np.random.rand()

    # Check that two consecutive numbers don't match
    cv.set_seed(default_seed)
    a = np.random.rand()
    b = np.random.rand()
    v = numba_rand()
    w = numba_rand()
    assert a != b
    assert v != w

    # Check that after resetting the seed, they do
    cv.set_seed(default_seed)
    c = np.random.rand()
    x = numba_rand()
    assert a == c
    assert v == x

    # Check that resetting with no argument, they don't again
    cv.set_seed()
    d = np.random.rand()
    y = numba_rand()
    cv.set_seed()
    e = np.random.rand()
    z = numba_rand()
    assert d != e
    assert y != z

    return a


def test_poisson():
    sc.heading('Poisson distribution')
    s1 = cv.poisson_test(10, 10)
    s2 = cv.poisson_test(10, 15)
    s3 = cv.poisson_test(0, 100)
    l1 = 1.0
    l2 = 0.05
    l3 = 1e-9
    assert s1 == l1
    assert s2 > l2
    assert s3 < l3
    print(f'Poisson assertions passed:')
    print(f'f(10,10) {s1} == {l1}')
    print(f'f(10,15) {s2} > {l2}')
    print(f'f(0,100) {s3} < {l3}')
    return s3


def test_samples(doplot=False):
    sc.heading('Samples distribution')

    n = 10000
    nbins = 40

    # Warning, must match utils.py!
    choices = [
        'uniform',
        'normal',
        'lognormal',
        'normal_pos',
        'normal_int',
        'lognormal_int',
        'neg_binomial'
        ]

    if doplot:
        pl.figure(figsize=(20,14))

    # Run the samples
    nchoices = len(choices)
    nsqr = np.ceil(np.sqrt(nchoices))
    results = {}
    for c,choice in enumerate(choices):
        if choice == 'neg_binomial':
            par1 = 10
            par2 = 0.5
        elif choice in ['lognormal', 'lognormal_int']:
            par1 = 1
            par2 = 0.5
        else:
            par1 = 0
            par2 = 5
        results[choice] = cv.sample(dist=choice, par1=par1, par2=par2, size=n)

        if doplot:
            pl.subplot(nsqr, nsqr, c+1)
            pl.hist(x=results[choice], bins=nbins)
            pl.title(f'dist={choice}, par1={par1}, par2={par2}')

    with pytest.raises(NotImplementedError):
        cv.sample(dist='not_found')

    return results



def test_choose():
    sc.heading('Choose people')
    x1 = cv.choose(10, 5)
    with pytest.raises(Exception):
        cv.choose_w(10, 5) # Requesting mroe people than are available
    print(f'Uniform sample from 0-9: {x1}')
    return x1


def test_choose_w():
    sc.heading('Choose weighted people')
    n = 100
    samples = 5
    lin = np.arange(n)
    lin = lin/lin.sum()
    x0 = cv.choose_w([0.01]*n, samples)
    x1 = cv.choose_w(lin, samples)
    x2 = cv.choose_w([1, 0, 0, 0, 0], 1)
    x3 = cv.choose_w([0.5, 0.5, 0, 0, 0], 1)
    assert x2[0] == 0
    assert x3[0] in [0,1]
    assert len(x0) == len(x1) == samples
    with pytest.raises(Exception):
        cv.choose_w([0.5, 0.5], 10) # Requesting mroe people than are available
    print(f'Uniform sample 0-99: x0 = {x0}, mean {x0.mean()}')
    print(f'Weighted sample 0-99: x1 = {x1}, mean {x1.mean()}')
    print(f'All weight on 0: x2 = {x2}')
    print(f'All weight on 0 or 1: x3 = {x3}')
    return x1


def test_doubling_time():

    sim = cv.Sim()
    sim.run(verbose=0)

    d = sc.objdict()

    # Test doubling time
    d.t1 = cv.get_doubling_time(sim, interval=[3,sim['n_days']+10], verbose=2) # should reset end date to sim['n_days']
    d.t2 = cv.get_doubling_time(sim, start_day=3,end_day=sim['n_days'])
    d.t3 = cv.get_doubling_time(sim, interval=[3,sim['n_days']], exp_approx=True)
    d.t4 = cv.get_doubling_time(sim, start_day=3, end_day=sim['n_days'], moving_window=4) # should return array
    d.t5 = cv.get_doubling_time(sim, series=np.power(1.03, range(100)), interval=[3,30], moving_window=3) # Should be a series with values = 23.44977..
    d.t6 = cv.get_doubling_time(sim, start_day=9, end_day=20, moving_window=1, series="cum_infections") # Should recast window to 2 then return a series with 100s in it
    with pytest.raises(ValueError):
        d.t7 = cv.get_doubling_time(sim, start_day=3, end_day=20, moving_window=4, series="cum_deaths") # Should fail, no growth in deaths

    print('NOTE: this test prints some warnings; these are intended.')
    return d


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    rnd1    = test_rand()
    rnd2    = test_poisson()
    samples = test_samples(doplot=doplot)
    people1 = test_choose()
    people2 = test_choose_w()
    dt = test_doubling_time()

    print('\n'*2)
    sc.toc()