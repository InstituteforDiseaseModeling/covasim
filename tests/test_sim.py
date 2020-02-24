'''
Simple example usage for the Covid-19 agent-based model
'''

#%% Imports and settings
import sciris as sc
import covid_abm

do_plot = False
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
    sim = test_sim(do_plot=do_plot, do_save=do_save)
    test_poisson()
    sc.toc()


print('Done.')