'''
Tests of the utilies for the model.
'''

#%% Imports and settings
import pytest
import sciris as sc
import covid_abm


#%% Define the tests

def test_poisson():
    s1 = covid_abm.poisson_test(10, 10)
    s2 = covid_abm.poisson_test(10, 15)
    s3 = covid_abm.poisson_test(0, 100)
    assert s1 == 1.0
    assert s2 > 0.05
    assert s3 < 1e-9
    print(f'Poisson assertions passed: p = {s1}, {s2}, {s3}')
    return
    

def test_choose_people():
    x1 = covid_abm.choose_people(10, 5)
    with pytest.raises(Exception):
        covid_abm.choose_people_weighted(10, 5) # Requesting mroe people than are available
    print(f'Choose people assertions passed: x1 = {x1}')
    return


def test_choose_people_weighted():
    x1 = covid_abm.choose_people_weighted([0.01]*100, 5)
    x2 = covid_abm.choose_people_weighted([1, 0, 0, 0, 0], 1)
    assert x2[0] == 0
    with pytest.raises(Exception):
        covid_abm.choose_people_weighted([0.5, 0, 0, 0, 0], 1) # Probabilities don't sum to 1
    with pytest.raises(Exception):
        covid_abm.choose_people_weighted([0.5, 0.5], 10) # Requesting mroe people than are available
    print(f'Choose weighted people assertions passed: x1 = {x1}, x2 = {x2}')
    return



#%% Run as a script
if __name__ == '__main__':
    sc.tic()
    test_poisson()
    test_choose_people()
    test_choose_people_weighted()
    sc.toc()


print('Done.')