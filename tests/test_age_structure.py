'''
Test different age structures
'''

#%% Imports and settings
import sciris as sc
import covasim as cv
import pytest


#%% Define the tests

def test_age_structure(): # If being run via pytest, turn off

    available = 'Lithuania'
    not_available = 'Ruritania'

    age_data = cv.data.loaders.get_age_distribution(available)

    with pytest.raises(ValueError):
        cv.data.loaders.get_age_distribution(not_available)

    return age_data



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    age_data = test_age_structure()

    sc.toc()


print('Done.')
