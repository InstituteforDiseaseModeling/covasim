'''
Tests of the utilies for the model.
'''

#%% Imports and settings
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
    

#%% Run as a script
if __name__ == '__main__':
    sc.tic()
    test_poisson()
    sc.toc()


print('Done.')