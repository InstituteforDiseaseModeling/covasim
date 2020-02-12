'''
Simple example usage for the Covid-19 agent-based model
'''

#%% Imports and settings
import sciris as sc
import covid_abm

do_plot = True
do_save = False


#%% Define the tests
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
    sim = test_sim(do_plot=do_plot, do_save=do_save)
    sc.toc()


print('Done.')

