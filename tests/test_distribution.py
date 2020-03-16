'''
Simple distribution tests
'''

#%% Imports and settings
import sciris as sc
import pylab as pl
from covasim.cova_base import utils as cov_ut
from covasim.cova_seattle import parameters as cov_pars

do_plot = 1
do_save = 0

#%% Define the tests

def test_distribution(config):
    val = cov_ut.sample(config)
    return val

def test_draws(subconfig, nTrials=100):
    draws = []
    for trial in range(nTrials):
        draw = cov_ut.sample(subconfig)
        draws.append(draw)

    fig, ax = pl.subplots()
    ax.hist(draws)

    return draws, ax


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    pars = cov_pars.make_pars()

    val = test_distribution(pars['incub'])
    print(f'Got val={val} from distribution')

    nTrials = 10000

    draws, ax = test_draws(pars['incub'], nTrials=nTrials)
    ax.set_title('Incubation Distrubion')
    pl.savefig('Incubation.png')

    draws, ax = test_draws(pars['dur'], nTrials=nTrials)
    ax.set_title('Infectious Duration')
    pl.savefig('Infectious.png')

    '''
    draws, ax = test_draws(config['clusterModel']['params']['beta'], nTrials=nTrials)
    ax.set_title('beta')
    pl.savefig('Beta.png')

    draws, ax = test_draws(config['clusterModel']['params']['N'], nTrials=nTrials)
    ax.set_title('N')
    pl.savefig('N.png')

    draws, ax = test_draws(config['branchingModel']['params']['branchingBeta'], nTrials=nTrials)
    ax.set_title('branchingBeta')
    pl.savefig('BranchingBeta.png')
    '''

    sc.toc()


print('Done.')
