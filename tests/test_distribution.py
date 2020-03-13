'''
Simple distribution tests
'''

#%% Imports and settings
import sciris as sc
import pylab as pl
from covid_abm import utils as cov_ut
from covid_seattle import parameters as cov_pars


from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

#%% Define the tests

def load_config(filename):
    with open(filename, 'r') as stream:
        config = load(stream, Loader=Loader)
    return config

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
