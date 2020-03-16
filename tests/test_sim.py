'''
Simple example usage for the Covid-19 agent-based model
'''

#%% Imports and settings
import pytest
import sciris as sc
import covasim.cova_seattle as cova

doplot = 1
do_save = 0


#%% Define the tests

def test_parsobj():
    sc.heading('Testing parameters object')

    pars1 = {'a':1, 'b':2}
    parsobj = cova.ParsObj(pars1)

    # Once created, you cannot directly add new keys to a parsobj, and a nonexistent key works like a dict
    with pytest.raises(KeyError): parsobj['c'] = 3
    with pytest.raises(KeyError): parsobj['c']

    # Only a dict is allowed
    with pytest.raises(TypeError):
        pars2 = ['a', 'b']
        cova.ParsObj(pars2)

    return parsobj


def test_sim(doplot=False, do_save=False): # If being run via pytest, turn off
    sc.heading('Basic sim test')

    # Settings
    seed = 1
    verbose = 1

    # Create and run the simulation
    sim = cova.Sim()
    sim.set_seed(seed)
    sim.run(verbose=verbose)

    # Optionally plot
    if doplot:
        sim.plot(do_save=do_save)

    return sim


def test_trans_tree(doplot=False, do_save=False): # If being run via pytest, turn off
    sc.heading('Transmission tree test')

    sim = cova.Sim() # Create the simulation
    sim.run(verbose=1) # Run the simulation
    if doplot:
        sim.plot(do_save=do_save)

    return sim.results['transtree']



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    parsobj = test_parsobj()
    sim     = test_sim(doplot=doplot, do_save=do_save)
    tt      = test_trans_tree(doplot=doplot)

    sc.toc()


print('Done.')
