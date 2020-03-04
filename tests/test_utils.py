'''
Tests of the utilies for the model.
'''

#%% Imports and settings
import pytest
import numpy as np
import numba as nb
import sciris as sc
import covid_abm


#%% Define the tests

def test_rand():
    
    default_seed = 1
    
    @nb.njit
    def numba_rand():
        return np.random.rand()
    
    # Check that two consecutive numbers don't match
    covid_abm.set_seed(default_seed)
    a = np.random.rand()
    b = np.random.rand()
    v = numba_rand()
    w = numba_rand()
    assert a != b 
    assert v != w
    
    # Check that after resetting the seed, they do
    covid_abm.set_seed(default_seed)
    c = np.random.rand()
    x = numba_rand()
    assert a == c 
    assert v == x
    
    # Check that resetting with no argument, they don't again
    covid_abm.set_seed() 
    d = np.random.rand()
    y = numba_rand()
    covid_abm.set_seed()
    e = np.random.rand()
    z = numba_rand()
    assert d != e 
    assert y != z
    
    return


def test_poisson():
    sc.heading('Poisson distribution')
    s1 = covid_abm.poisson_test(10, 10)
    s2 = covid_abm.poisson_test(10, 15)
    s3 = covid_abm.poisson_test(0, 100)
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
    return
    

def test_choose_people():
    sc.heading('Choose people')
    x1 = covid_abm.choose_people(10, 5)
    with pytest.raises(Exception):
        covid_abm.choose_people_weighted(10, 5) # Requesting mroe people than are available
    print(f'Uniform sample from 0-9: {x1}')
    return


def test_choose_people_weighted():
    sc.heading('Choose weighted people')
    n = 100
    samples = 5
    lin = np.arange(n)
    lin = lin/lin.sum()
    x0 = covid_abm.choose_people_weighted([0.01]*n, samples)
    x1 = covid_abm.choose_people_weighted(lin, samples)
    x2 = covid_abm.choose_people_weighted([1, 0, 0, 0, 0], 1)
    x3 = covid_abm.choose_people_weighted([0.5, 0.5, 0, 0, 0], 1)
    assert x2[0] == 0
    assert x3[0] in [0,1]
    assert len(x0) == len(x1) == samples
    with pytest.raises(Exception):
        covid_abm.choose_people_weighted([0.5, 0, 0, 0, 0], 1) # Probabilities don't sum to 1
    with pytest.raises(Exception):
        covid_abm.choose_people_weighted([0.5, 0.5], 10) # Requesting mroe people than are available
    print(f'Uniform sample 0-99: x0 = {x0}, mean {x0.mean()}')
    print(f'Weighted sample 0-99: x1 = {x1}, mean {x1.mean()}')
    print(f'All weight on 0: x2 = {x2}')
    print(f'All weight on 0 or 1: x3 = {x3}')
    return



#%% Run as a script
if __name__ == '__main__':
    sc.tic()
    
    test_rand()
    test_poisson()
    test_choose_people()
    test_choose_people_weighted()
    
    print('\n'*2)
    sc.toc()
    print('Done.')