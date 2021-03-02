import covasim as cv
import covasim.defaults as cvd
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np


do_plot   = 1
do_show   = 0
do_save   = 1


def test_basic_reinfection(do_plot=False, do_show=True, do_save=False):
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
          'pars': {
              'imm_pars': {k: dict(form='exp_decay', pars={'init_val': 1., 'half_life': None}) for k in cvd.immunity_axes}
          }
        },
        'med_halflife': {
          'name':'3 month waning susceptibility',
          'pars': {
              'imm_pars': {k: dict(form='exp_decay', pars={'init_val': 1., 'half_life': 90}) for k in
                               cvd.immunity_axes}
          }
        },
        'med_halflife_bysev': {
          'name':'2 month waning susceptibility for symptomatics only',
          'pars': {
              'rel_imm': {'asymptomatic': 0},
              'imm_pars': {k: dict(form='exp_decay', pars={'init_val': 1., 'half_life': 60}) for k in
                               cvd.immunity_axes}
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

    return scens


def test_strainduration(do_plot=False, do_show=True, do_save=False):
    sc.heading('Run a sim with 2 strains, one of which has a much longer period before symptoms develop')
    sc.heading('Setting up...')

    imported_strain = {'dur':{'inf2sym': {'dist': 'lognormal_int', 'par1': 10.0, 'par2': 0.9}}}
    imports = cv.import_strain(strain=imported_strain, days=10, n_imports=30)
    tp = cv.test_prob(symp_prob=0.2) # Add an efficient testing program

    base_pars = {
        'beta': 0.015, # Make beta higher than usual so people get infected quickly
        'n_days': 120,
    }

    n_runs = 1
    base_sim = cv.Sim(base_pars)

    # Define the scenarios
    scenarios = {
        'baseline': {
          'name':'1 day to symptoms',
          'pars': {'interventions': [tp]}
        },
        'slowsymp': {
          'name':'10 days to symptoms',
          'pars': {'interventions': [imports, tp]}
        }
    }

    metapars = {'n_runs': n_runs}
    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run()

    to_plot = sc.objdict({
        'New infections': ['new_infections'],
        'Cumulative infections': ['cum_infections'],
        'New diagnoses': ['new_diagnoses'],
        'Cumulative diagnoses': ['cum_diagnoses'],
    })
    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path=f'results/test_strainduration.png', to_plot=to_plot)

    return scens


def test_import1strain(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test introducing a new strain partway through a sim')
    sc.heading('Setting up...')

    strain_labels = [
        'Strain 1',
        'Strain 2: 1.5x more transmissible'
    ]

    imported_strain = {
        'rel_beta': 1.5,
#        'imm_pars': {k: dict(form='exp_decay', pars={'init_val':1., 'half_life':100}) for k in cvd.immunity_axes}
        'imm_pars': {k: dict(form='logistic_decay', pars={'init_val': 1., 'half_val': 10, 'lower_asymp': 0.1, 'decay_rate': -5}) for k in cvd.immunity_axes}
    }

    imports = cv.import_strain(strain=imported_strain, days=1, n_imports=30)
    sim = cv.Sim(interventions=imports, label='With imported infections')
    sim.run()

    if do_plot:
        plot_results(sim, key='incidence_by_strain', title='Imported strain on day 30 (cross immunity)', filename='test_importstrain1', labels=strain_labels, do_show=do_show, do_save=do_save)
        plot_shares(sim, key='new_infections', title='Shares of new infections by strain', filename='test_importstrain1_shares', do_show=do_show, do_save=do_save)
    return sim


def test_import2strains(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test introducing 2 new strains partway through a sim')
    sc.heading('Setting up...')

    strain2 = cv.make_strain('b117')
    strain3 = cv.make_strain('sa variant')

    imports = [cv.import_strain(strain=strain2, days=10, n_imports=20),
               cv.import_strain(strain=strain3, days=30, n_imports=20),
               ]
    sim = cv.Sim(interventions=imports, label='With imported infections')
    sim.run()

    strain_labels = [
        'Strain 1: Wild Type',
        'Strain 2: UK Variant on day 10',
        'Strain 3: SA Variant on day 30'
    ]

    if do_plot:
        plot_results(sim, key='incidence_by_strain', title='Imported strains', filename='test_importstrain2', labels=strain_labels, do_show=do_show, do_save=do_save)
        plot_shares(sim, key='new_infections', title='Shares of new infections by strain',
                filename='test_importstrain2_shares', do_show=do_show, do_save=do_save)
    return sim


def test_importstrain_longerdur(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test introducing a new strain with longer duration partway through a sim')
    sc.heading('Setting up...')

    strain_labels = [
        'Strain 1: beta 0.016',
        'Strain 2: beta 0.025'
    ]

    pars = {
        'n_days': 120,
    }

    imported_strain = {
        'rel_beta': 1.5,
        'dur': {'exp2inf':dict(dist='lognormal_int', par1=6.0,  par2=2.0)}
    }

    imports = cv.import_strain(strain=imported_strain, days=10, n_imports=30)
    sim = cv.Sim(pars=pars, interventions=imports, label='With imported infections')
    sim.run()

    if do_plot:
        plot_results(sim, key='incidence_by_strain', title='Imported strain on day 30 (longer duration)', filename='test_importstrain1', labels=strain_labels, do_show=do_show, do_save=do_save)
        plot_shares(sim, key='new_infections', title='Shares of new infections by strain', filename='test_importstrain_longerdur_shares', do_show=do_show, do_save=do_save)
    return sim


def test_import2strains_changebeta(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test introducing 2 new strains partway through a sim, with a change_beta intervention')
    sc.heading('Setting up...')

    strain2 = {'rel_beta': 1.5,
               'rel_severe_prob': 1.3,
               }

    strain3 = {
        'rel_beta': 2,
        'rel_symp_prob': 1.6,
    }

    intervs  = [
        cv.change_beta(days=[5, 20, 40], changes=[0.8, 0.7, 0.6]),
        cv.import_strain(strain=strain2, days=10, n_imports=20),
        cv.import_strain(strain=strain3, days=30, n_imports=20),
               ]
    sim = cv.Sim(interventions=intervs, label='With imported infections')
    sim.run()

    strain_labels = [
        'Strain 1: beta 0.016',
        'Strain 2: beta 0.025, 20 imported day 10',
        'Strain 3: beta 0.05, 20 imported day 30'
    ]

    if do_plot:
        plot_results(sim, key='incidence_by_strain', title='Imported strains', filename='test_importstrain2', labels=strain_labels, do_show=do_show, do_save=do_save)
        plot_shares(sim, key='new_infections', title='Shares of new infections by strain',
                filename='test_importstrain2_shares', do_show=do_show, do_save=do_save)
    return sim


def plot_results(sim, key, title, filename=None, do_show=True, do_save=False, labels=None):

    results = sim.results
    results_to_plot = results[key]

    # extract data for plotting
    x = sim.results['t']
    y = results_to_plot.values
    y = np.transpose(y)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='Day of simulation', ylabel=results_to_plot.name, title=title)

    if labels is None:
        labels = [0]*len(y[0])
        for strain in range(len(y[0])):
            labels[strain] = f'Strain {strain +1}'
    ax.legend(labels)

    if do_show:
        plt.show()
    if do_save:
        cv.savefig(f'results/{filename}.png')

    return


def plot_shares(sim, key, title, filename=None, do_show=True, do_save=False, labels=None):

    results = sim.results
    n_strains = sim.results['new_infections_by_strain'].values.shape[0] # TODO: this should be stored in the sim somewhere more intuitive!
    prop_new = {f'Strain {s}': sc.safedivide(results[key+'_by_strain'].values[s,:], results[key].values, 0) for s in range(n_strains)}
    num_new = {f'Strain {s}': results[key+'_by_strain'].values[s,:] for s in range(n_strains)}

    # extract data for plotting
    x = sim.results['t']
    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].stackplot(x, prop_new.values(),
                 labels=prop_new.keys())
    ax[0].legend(loc='upper left')
    ax[0].set_title(title)
    ax[1].stackplot(sim.results['t'], num_new.values(),
                 labels=num_new.keys())
    ax[1].legend(loc='upper left')
    ax[1].set_title(title)

    if do_show:
        plt.show()
    if do_save:
        cv.savefig(f'results/{filename}.png')

    return


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    # Run simplest possible test
    # if 0:
    #     sim = cv.Sim()
    #     sim.run()

    # Run more complex tests
    sim1 = test_import1strain(do_plot=do_plot, do_save=do_save, do_show=do_show)
#    sim2 = test_import2strains(do_plot=do_plot, do_save=do_save, do_show=do_show)
#    sim3 = test_importstrain_longerdur(do_plot=do_plot, do_save=do_save, do_show=do_show)

    # TODO: the next test isn't working, need to check change_beta logic
    # sim4 = test_import2strains_changebeta(do_plot=do_plot, do_save=do_save, do_show=do_show)

    # TODO: the next two tests aren't working, need to check scenario logic
    # scens1 = test_basic_reinfection(do_plot=do_plot, do_save=do_save, do_show=do_show)
    # scens2 = test_strainduration(do_plot=do_plot, do_save=do_save, do_show=do_show)

    # The next tests are deprecated, can be removed
    # simX = test_importstrain_args()
    # p1, p2, p3 = test_par_refactor()

    sc.toc()


print('Done.')



# DEPRECATED
# def test_importstrain_args():
#     sc.heading('Test flexibility of arguments for the import strain "intervention"')
#
#     # Run sim with a single strain initially, then introduce a new strain that's more transmissible on day 10
#     immunity = [
#         {'init_immunity': 1., 'half_life': 180, 'cross_factor': 1},
#     ]
#     pars = {
#         'n_strains': 1,
#         'beta': [0.016],
#         'immunity': immunity
#     }
#
#     # All these should run
#     imports = cv.import_strain(days=50, beta=0.03)
#     #imports = cv.import_strain(days=[10, 50], beta=0.03)
#     #imports = cv.import_strain(days=50, beta=[0.03, 0.05])
#     #imports = cv.import_strain(days=[10, 20], beta=[0.03, 0.05])
#     #imports = cv.import_strain(days=50, beta=[0.03, 0.05, 0.06])
#     #imports = cv.import_strain(days=[10, 20], n_imports=[5, 10], beta=[0.03, 0.05], init_immunity=[1, 1],
#     #                          half_life=[180, 180], cross_factor=[0, 0])
#     #imports = cv.import_strain(days=[10, 50], beta=0.03, cross_factor=[0.4, 0.6])
#     #imports = cv.import_strain(days=['2020-04-01', '2020-05-01'], beta=0.03)
#
#     # This should fail
#     #imports = cv.import_strain(days=[20, 50], beta=[0.03, 0.05, 0.06])
#
#     sim = cv.Sim(pars=pars, interventions=imports)
#     sim.run()
#
#
#     return sim
#

# def test_par_refactor():
#     '''
#     The purpose of this test is to experiment with different representations of the parameter structures
#     Still WIP!
#     '''
#
#     # Simplest case: add a strain to beta
#     p1 = cv.Par(name='beta', val=0.016, by_strain=True)
#     print(p1.val) # Prints all the stored values of beta
#     print(p1[0])  # Can index beta like an array to pull out strain-specific values
#     p1.add_strain(new_val = 0.025)
#
#     # Complex case: add a strain that's differentiated by severity for kids 0-20
#     p2 = cv.Par(name='sus_ORs', val=np.array([0.34, 0.67, 1., 1., 1., 1., 1.24, 1.47, 1.47, 1.47]), by_strain=True, by_age=True)
#     print(p2.val) # Prints all the stored values for the original strain
#     print(p2[0])  # Can index beta like an array to pull out strain-specific values
#     p2.add_strain(new_val=np.array([1., 1., 1., 1., 1., 1., 1.24, 1.47, 1.47, 1.47]))
#
#     # Complex case: add a strain that's differentiated by duration of disease
#     p3 = cv.Par(name='dur_asym2rec', val=dict(dist='lognormal_int', par1=8.0,  par2=2.0), by_strain=True, is_dist=True)
#     print(p3.val) # Prints all the stored values for the original strain
#     print(p3[0])  # Can index beta like an array to pull out strain-specific values
#     p3.add_strain(new_val=dict(dist='lognormal_int', par1=12.0,  par2=2.0))
#     p3.get(strain=1, n=6)
#
#     return p1, p2, p3

