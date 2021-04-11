'''
Tests for immune waning, strains, and vaccine intervention.
'''

#%% Imports and settings
# import pytest
import sciris as sc
import covasim as cv

do_plot = 1
do_save = 0
cv.options.set(interactive=False) # Assume not running interactively

base_pars = dict(
    pop_size = 1e3,
    verbose = -1,
)

#%% Define the tests

def test_waning(do_plot=False):
    sc.heading('Testing with and without waning')
    s1 = cv.Sim(base_pars, n_days=300, use_waning=True, label='No waning')
    s2 = cv.Sim(base_pars, n_days=300, use_waning=True, label='With waning')
    msim = cv.MultiSim([s1,s2])
    msim.run()
    if do_plot:
        msim.plot('strain')
    return msim


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    msim1 = test_waning(do_plot=do_plot)

    sc.toc(T)
    print('Done.')
