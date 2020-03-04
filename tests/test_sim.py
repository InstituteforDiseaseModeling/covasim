'''
Simple example usage for the Covid-19 agent-based model
'''

#%% Imports and settings
import pytest
import sciris as sc
import covid_abm

do_plot = 0
do_save = 0


#%% Define the tests

def test_parsobj():
    pars1 = {'a':1, 'b':2}
    parsobj = covid_abm.ParsObj(pars1)
    
    # Once created, you cannot directly add new keys to a parsobj, and a nonexistent key works like a dict
    with pytest.raises(KeyError): parsobj['c'] = 3
    with pytest.raises(KeyError): parsobj['c']
    
    # Only a dict is allowed
    with pytest.raises(TypeError):
        pars2 = ['a', 'b']
        covid_abm.ParsObj(pars2)
    
    return parsobj


def test_sim(do_plot=False, do_save=False): # If being run via pytest, turn off

    # Create the simulation
    sim = covid_abm.Sim()

    # Run the simulation
    sim.run()

    # Optionally plot
    if do_plot:
        sim.plot(do_save=do_save)

    return sim


#%% Run as a script
if __name__ == '__main__':
    sc.tic()
    parsobj = test_parsobj()
    # sim     = test_sim(do_plot=do_plot, do_save=do_save)
    sc.toc()


print('Done.')
