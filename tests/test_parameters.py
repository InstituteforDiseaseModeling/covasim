'''
Test that the parameters and data files are being created correctly.
'''

#%% Imports
import os
import pytest
import sciris as sc
import covasim as cv # NOTE: this is the only tests script that doesn't use base

do_plot = False


#%% Define the tests
def test_parameters():
    sc.heading('Model parameters')
    pars = cv.make_pars()
    sc.pp(pars)
    return pars


def test_data():
    sc.heading('Data loading')
    data = cv.load_data(os.path.join(sc.thisdir(__file__), 'example_data.csv'))
    sc.pp(data)

    # Check that it is looking for the right file
    with pytest.raises(FileNotFoundError):
        data = cv.load_data(filename='file_not_found.csv')

    return data


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    pars = test_parameters()
    data = test_data()

    sc.toc()


print('Done.')
