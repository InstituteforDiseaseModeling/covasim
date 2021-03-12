import covasim as cv
import covasim.defaults as cvd
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np


plot_args = dict(do_plot=1, do_show=0, do_save=1)

def test_reinfection_scens(do_plot=False, do_show=True, do_save=False):
    sc.heading('Run a basic sim with 1 strain, varying reinfection risk')
    sc.heading('Setting up...')

    # Define baseline parameters
    base_pars = {
        'beta': 0.1, # Make beta higher than usual so people get infected quickly
        'n_days': 240,
    }

    n_runs = 3
    base_sim = cv.Sim(base_pars)

    # Define the scenarios
    scenarios = {
        'baseline': {
            'name':'No reinfection',
            'pars': {'imm_pars': {k: dict(form='exp_decay', pars={'init_val': 1., 'half_life': None}) for k in cvd.immunity_axes},
                     'rel_imm': {k: 1 for k in cvd.immunity_sources}
                     },
        },
        'med_halflife': {
            'name':'3 month waning susceptibility',
            'pars': {'imm_pars': {k: dict(form='exp_decay', pars={'init_val': 1., 'half_life': 90}) for k in cvd.immunity_axes}},
        },
        'med_halflife_bysev': {
            'name':'2 month waning susceptibility for symptomatics only',
            'pars': {'rel_imm': {'asymptomatic': 0, 'mild': 1, 'severe': 1},
                     'imm_pars': {k: dict(form='exp_decay', pars={'init_val': 1., 'half_life': 60}) for k in cvd.immunity_axes}
            }
        },
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

    return sim


def test_reinfection(do_plot=False, do_show=True, do_save=False):
    sc.heading('Run a basic sim with 1 strain and no reinfections')
    sc.heading('Setting up...')

    pars = {
        'beta': 0.015,
        'n_days': 120,
        'rel_imm': {k:1 for k in cvd.immunity_sources}
    }
    sim = cv.Sim(pars=pars)
    sim.run()

    to_plot = sc.objdict({
        'New infections': ['new_infections'],
        'Cumulative infections': ['cum_infections'],
        'New reinfections': ['new_reinfections'],
    })
    if do_plot:
        sim.plot(do_save=do_save, do_show=do_show, fig_path=f'results/test_reinfection.png', to_plot=to_plot)

    return sim



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    # Run simplest possible test
    if 0:
        sim = cv.Sim()
        sim.run()

    # Run more complex tests
    sim1 = test_reinfection(**plot_args)
    #scens1 = test_reinfection_scens(**plot_args)

    sc.toc()


print('Done.')

