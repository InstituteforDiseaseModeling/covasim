'''
Test that the parameters and data files are being created correctly.
'''

#%% Imports
import pytest
import sciris as sc
from covid_abm import parameters as cov_pars


#%% Define the tests
def test_parameters():
    sc.heading('Model parameters')
    pars = cov_pars.make_pars()
    sc.pp(pars)
    return pars


def test_data():
    sc.heading('Data loading')
    data = cov_pars.load_data()
    sc.pp(data)
    
    # Check that it is looking for the right file
    with pytest.raises(FileNotFoundError):
        data = cov_pars.load_data(filename='file_not_found.csv')
    
    return data

#%% Run as a script
if __name__ == '__main__':
    sc.tic()
    pars = test_parameters()
    data = test_data()
    sc.toc()


print('Done.')