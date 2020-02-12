'''
Simple example usage for the Covid-19 agent-based model
'''

#%% Imports and settings
import sciris as sc
import covid_abm

doplot = True
dosave = False


#%% Define the tests
def test_sim(doplot=False, dosave=False): # If being run via pytest, turn off
    sim = covid_abm.Sim()
    sim.run()
    if doplot:
        sim.plot(dosave=dosave)


#%% Run as a script
if __name__ == '__main__':
    sc.tic()
    sim = test_sim(dosave=dosave)
    sc.toc()


print('Done.')

