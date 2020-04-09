'''
Test different population options
'''

#%% Imports and settings
import pytest
import pylab as pl
import sciris as sc
import covasim as cv


doplot = 1


#%% Define the tests

def test_pop_options(doplot=False): # If being run via pytest, turn off
    sc.heading('Basic populations tests')

    # popchoices = ['microstructure', 'random']
    popchoices = ['random', 'microstructure']

    basepars = {
        'n': 5000,
        'n_infected': 10,
        'n_days': 30
        }

    sims = sc.objdict()
    for popchoice in popchoices:
        sc.heading(f'Running {popchoice}')
        sims[popchoice] = cv.Sim()
        sims[popchoice].update_pars(basepars)
        sims[popchoice]['usepopdata'] = popchoice
        sims[popchoice].run()

    if doplot:
        for key,sim in sims.items():
            sim.plot()
            try:
                pl.gcf().axes[0].set_title(f'Counts: {key}')
            except:
                pass

    return sims



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    sims1 = test_pop_options(doplot=doplot)

    sc.toc()


print('Done.')
