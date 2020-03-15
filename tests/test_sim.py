'''
Simple example usage for the Covid-19 agent-based model
'''

#%% Imports and settings
import pytest
import sciris as sc
from covasim import cova_seattle as covid

doplot = 1
do_save = 0


#%% Define the tests

def test_parsobj():
    pars1 = {'a':1, 'b':2}
    parsobj = covid.ParsObj(pars1)
    
    # Once created, you cannot directly add new keys to a parsobj, and a nonexistent key works like a dict
    with pytest.raises(KeyError): parsobj['c'] = 3
    with pytest.raises(KeyError): parsobj['c']
    
    # Only a dict is allowed
    with pytest.raises(TypeError):
        pars2 = ['a', 'b']
        covid.ParsObj(pars2)
    
    return parsobj


def test_sim(doplot=False, do_save=False): # If being run via pytest, turn off
    
    # Settings
    seed = 1
    verbose = 1

    # Create and run the simulation
    sim = covid.Sim() # TODO: reconcile with covid
    sim.set_seed(seed)
    sim.run(verbose=verbose)

    # Optionally plot
    if doplot:
        sim.plot(do_save=do_save)

    return sim


def test_trans_tree(doplot=False, do_save=False): # If being run via pytest, turn off

    sim = covid.Sim() # Create the simulation
    sim.run(verbose=1) # Run the simulation
    if doplot:
        sim.plot(do_save=do_save)

    return sim.results['transtree']



def test_multiscale(doplot=False, do_save=False): # If being run via pytest, turn off
    
    sim1 = covid.Sim() # Create the simulation
    sim1['n'] = 1000
    sim1['n_days'] = 20
    sim.run(verbose=1) # Run the simulation
    if doplot:
        sim.plot(do_save=do_save)

    return sim.results['transtree']


#%% Run as a script
if __name__ == '__main__':
    sc.tic()
    # parsobj = test_parsobj()
    sim     = test_sim(doplot=doplot, do_save=do_save)
    # trans_tree = test_trans_tree(doplot=doplot)
    # sim = test_multiscale(doplot=doplot)
    sc.toc()


print('Done.')
