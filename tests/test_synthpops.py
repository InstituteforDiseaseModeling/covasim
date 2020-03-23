'''
Test different population options
'''

#%% Imports and settings
import pylab as pl
import sciris as sc
import covasim as cova

doplot = 1


#%% Define the tests

def test_import():
    sc.heading('Testing imports')

    assert cova._requirements.available['synthpops'] == True
    import synthpops as sp
    print(sp.datadir)

    return


def test_pop_options(doplot=False): # If being run via pytest, turn off
    sc.heading('Basic populations tests')

    popchoices = ['random', 'bayesian', 'data']

    basepars = {
        'n': 3000,
        'n_infected': 10,
        'contacts': 20,
        'n_days': 90
        }

    sims = sc.objdict()
    for popchoice in popchoices:
        sc.heading(f'Running {popchoice}')
        sims[popchoice] = cova.Sim()
        sims[popchoice].update_pars(basepars)
        sims[popchoice]['usepopdata'] = popchoice
        sims[popchoice].run()

    if doplot:
        for key,sim in sims.items():
            sim.plot()
            pl.gcf().axes[0].set_title(f'Counts: {key}')

    return sims



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    test_import()
    sims = test_pop_options(doplot=doplot)

    sc.toc()


print('Done.')
