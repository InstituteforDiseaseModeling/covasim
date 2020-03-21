'''
Simple example usage for the Covid-19 agent-based model
'''

#%% Imports and settings
import pylab as pl
import sciris as sc
import covasim as cova

doplot = 1
figsize = (20,16)


#%% Define the tests

def test_age_structure(doplot=False): # If being run via pytest, turn off
    pass # Not implemented yet

    # # Create and run the simulation without age structure
    # sims = sc.objdict()

    # sims.without = cova.Sim()
    # sims.without['usepopdata'] = 0
    # sims.without.run(verbose=1)

    # # ...and with
    # sims.withdata = cova.Sim()
    # sims.withdata['usepopdata'] = 1
    # sims.withdata.run(verbose=1)

    # # Calculate ages
    # ages = sc.objdict()
    # for key in sims.keys():
    #     ages[key] = []
    #     for person in sims[key].people.values():
    #         ages[key].append(person.age)

    # # Optionally plot
    # if doplot:
    #     nbins = 100

    #     # Plot epi results
    #     for sim in sims.values():
    #         sim.plot()

    #     # Plot ages
    #     pl.figure(figsize=figsize)
    #     for a,key,age in ages.enumitems():
    #         if key == 'without':
    #             title = f'Normally-distributed age distribution'
    #         elif key == 'withdata':
    #             title = f'Age distribution based on Seattle census data'
    #         pl.subplot(2,1,a+1)
    #         pl.hist(ages[a], nbins)
    #         pl.title(title)
    #         pl.xlabel('Age')
    #         pl.ylabel('Number of people')

    # return sims, ages


#%% Run as a script
if __name__ == '__main__':
    sc.tic()
    print('Not implemented yet')
    # sims, ages = test_age_structure(doplot=doplot)
    sc.toc()


print('Done.')
