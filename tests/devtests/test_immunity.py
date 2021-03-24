import covasim as cv
import covasim.defaults as cvd
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np


plot_args = dict(do_plot=1, do_show=1, do_save=1)

def test_reinfection_scens(do_plot=False, do_show=True, do_save=False):
    sc.heading('Run a basic sim with varying reinfection risk')
    sc.heading('Setting up...')

    # Define baseline parameters
    base_pars = {
        'beta': 0.1, # Make beta higher than usual so people get infected quickly
        'n_days': 350,
    }

    n_runs = 3
    base_sim = cv.Sim(base_pars)
    b1351 = cv.Strain('b1351', days=100, n_imports=20)

    # Define the scenarios
    scenarios = {
        # 'baseline': {
        #     'name':'Baseline',
        #     'pars': {'NAb_decay': dict(form='nab_decay', pars={'init_decay_rate': np.log(2)/90, 'init_decay_time': 250, 'decay_decay_rate': 0.001})},
        # },
        # 'slower': {
        #     'name':'Slower',
        #     'pars': {'NAb_decay': dict(form='nab_decay', pars={'init_decay_rate': np.log(2)/120, 'init_decay_time': 250, 'decay_decay_rate': 0.001})},
        # },
        # 'faster': {
        #     'name':'Faster',
        #     'pars': {'NAb_decay': dict(form='nab_decay', pars={'init_decay_rate': np.log(2)/30, 'init_decay_time': 150, 'decay_decay_rate': 0.01})},
        # },
        'baseline_b1351': {
            'name': 'Baseline, B1351 on day 40',
            'pars': {'strains': [b1351]},

        },
        'slower_b1351': {
            'name': 'Slower, B1351 on day 40',
            'pars': {'NAb_decay': dict(form='nab_decay',
                                       pars={'init_decay_rate': np.log(2) / 120, 'init_decay_time': 250,
                                             'decay_decay_rate': 0.001}),
                     'strains': [b1351]},
        },
        'faster_b1351': {
            'name': 'Faster, B1351 on day 40',
            'pars': {'NAb_decay': dict(form='nab_decay',
                                       pars={'init_decay_rate': np.log(2) / 30, 'init_decay_time': 150,
                                             'decay_decay_rate': 0.01}),
                     'strains': [b1351]},
        },
        # 'even_faster_b1351': {
        #     'name': 'Even faster, B1351 on day 40',
        #     'pars': {'NAb_decay': dict(form='nab_decay',
        #                                pars={'init_decay_rate': np.log(2) / 30, 'init_decay_time': 50,
        #                                      'decay_decay_rate': 0.1}),
        #              'strains': [b1351]},
        # },
    }

    metapars = {'n_runs': n_runs}
    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run()

    to_plot = sc.objdict({
        'New infections': ['new_infections'],
        'Cumulative infections': ['cum_infections'],
        'New reinfections': ['new_reinfections'],
        # 'Cumulative reinfections': ['cum_reinfections'],
    })
    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path=f'results/test_basic_reinfection.png', to_plot=to_plot)

    return scens




#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    # Run simplest possible test
    # if 0:
    #     sim = cv.Sim()
    #     sim.run()

    # Run more complex tests
    scens1 = test_reinfection_scens(**plot_args)

    sc.toc()


print('Done.')

