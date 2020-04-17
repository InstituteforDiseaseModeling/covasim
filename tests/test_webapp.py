'''
Simple example usage for the Covid-19 agent-based model
'''

#%% Imports and settings
import sciris as sc
import covasim.webapp as cw
import pprint
from typing import Dict

def has_errs(output:Dict = None):
    return output and output['errs'] and len(output['errs'])

#%% Define the tests


def test_get_defaults():
    sc.heading('Testing parameters')

    pars = cw.get_defaults()

    return pars


def test_run_sim():
    sc.heading('Testing webapp')

    pars = cw.get_defaults()
    output = cw.run_sim(sim_pars=pars['sim_pars'], epi_pars=pars['epi_pars'])
    if has_errs(output):
        errormsg = 'Webapp encountered an error:\n'
        errormsg += pprint.pformat(output['errs'], indent=2)
        raise Exception(errormsg)
    return output

def test_run_sim_invalid_pars():
    output = cw.run_sim(sim_pars='invalid', epi_pars='invalid')
    if not has_errs(output):
        errormsg = 'Invalid parameters failed to raise an error'
        raise Exception(errormsg)

    return output


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    pars = test_pars()
    output = test_webapp()

    sc.toc()


print('Done.')
