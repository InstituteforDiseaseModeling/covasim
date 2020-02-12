'''
Test that the parameters and data files are being created correctly.
'''

#%% Imports
import pytest
import sciris as sc
import covid_abm


#%% Define the tests
def test_parameters():
    sc.heading('Model parameters')
    pars = covid_abm.make_pars()
    sc.pp(pars)
    return pars


def test_data():
    sc.heading('Data loading')
    data = covid_abm.load_data()
    sc.pp(data)
    
    # Check that it is looking for the right file
    with pytest.raises(FileNotFoundError):
        data = covid_abm.load_data(filename='file_not_found.csv')
    
    return data

#%% Run as a script
if __name__ == '__main__':
    sc.tic()
    pars = test_parameters()
    data = test_data()
    sc.toc()


print('Done.')