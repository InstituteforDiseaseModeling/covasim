'''
Test the webapp
'''

#%% Imports and settings
import sciris as sc
import covasim.webapp as cw


#%% Define the tests

def test_pars():
    sc.heading('Testing parameters')

    pars = cw.get_defaults()

    return pars


def test_webapp():
    sc.heading('Testing webapp')

    pars = cw.get_defaults()
    output = cw.run_sim(sim_pars=pars['sim_pars'], epi_pars=pars['epi_pars'])
    if output['errs']:
        errormsg = 'Webapp encountered an error:\n'
        errormsg += sc.pp(str(output['errs']), doprint=False)
        raise Exception(errormsg)

    cw.run_sim(sim_pars='invalid', epi_pars='invalid')
    if not output['errs']:
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
