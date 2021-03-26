import covasim as cv
import covasim.defaults as cvd
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


do_plot   = 1
do_show   = 0
do_save   = 1


def test_import1strain(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test introducing a new strain partway through a sim')
    sc.heading('Setting up...')

    strain_labels = [
        'Strain 1',
        'Strain 2: 1.5x more transmissible'
    ]

    strain_pars = {
        'rel_beta': 1.5,
    }
    pars = {
        'beta': 0.01
    }
    strain = cv.Strain(strain_pars, days=1, n_imports=20)
    sim = cv.Sim(pars=pars, strains=strain, analyzers=cv.snapshot(30, 60))
    sim.run()

    if do_plot:
        plot_results(sim, key='incidence_by_strain', title='Imported strain on day 30 (cross immunity)', filename='test_importstrain1', labels=strain_labels, do_show=do_show, do_save=do_save)
        plot_shares(sim, key='new_infections', title='Shares of new infections by strain', filename='test_importstrain1_shares', do_show=do_show, do_save=do_save)

    return sim


def test_import2strains(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test introducing 2 new strains partway through a sim')
    sc.heading('Setting up...')

    b117 = cv.Strain('b117', days=1, n_imports=20)
    p1 = cv.Strain('sa variant', days=2, n_imports=20)
    sim = cv.Sim(strains=[b117, p1], label='With imported infections')
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

    strain_pars = {
        'rel_beta': 1.5,
        'dur': {'exp2inf':dict(dist='lognormal_int', par1=6.0,  par2=2.0)}
    }

    strain = cv.Strain(strain=strain_pars, strain_label='Custom strain', days=10, n_imports=30)
    sim = cv.Sim(pars=pars, strains=strain, label='With imported infections')
    sim.run()

    if do_plot:
        plot_results(sim, key='incidence_by_strain', title='Imported strain on day 30 (longer duration)', filename='test_importstrain1', labels=strain_labels, do_show=do_show, do_save=do_save)
        plot_shares(sim, key='new_infections', title='Shares of new infections by strain', filename='test_importstrain_longerdur_shares', do_show=do_show, do_save=do_save)
    return sim


def test_import2strains_changebeta(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test introducing 2 new strains partway through a sim, with a change_beta intervention')
    sc.heading('Setting up...')

    strain2 = {'rel_beta': 1.5,
               'rel_severe_prob': 1.3}

    strain3 = {'rel_beta': 2,
               'rel_symp_prob': 1.6}

    intervs  = cv.change_beta(days=[5, 20, 40], changes=[0.8, 0.7, 0.6])
    strains  = [cv.Strain(strain=strain2, days=10, n_imports=20),
                cv.Strain(strain=strain3, days=30, n_imports=20),
               ]
    sim = cv.Sim(interventions=intervs, strains=strains, label='With imported infections')
    sim.run()

    strain_labels = [
        f'Strain 1: beta {sim["beta"]}',
        f'Strain 2: beta {sim["beta"]*sim["rel_beta"][1]}, 20 imported day 10',
        f'Strain 3: beta {sim["beta"]*sim["rel_beta"][2]}, 20 imported day 30'
    ]

    if do_plot:
        plot_results(sim, key='incidence_by_strain', title='Imported strains', filename='test_import2strains_changebeta', labels=strain_labels, do_show=do_show, do_save=do_save)
        plot_shares(sim, key='new_infections', title='Shares of new infections by strain',
                filename='test_import2strains_changebeta_shares', do_show=do_show, do_save=do_save)
    return sim



#%% Vaccination tests

def test_vaccine_1strain(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test vaccination with a single strain')
    sc.heading('Setting up...')

    pars = {
        'beta': 0.015,
        'n_days': 120,
    }

    pfizer = cv.vaccinate(days=[20], vaccine_pars='pfizer')
    sim = cv.Sim(
        pars=pars,
        interventions=pfizer
    )
    sim.run()

    to_plot = sc.objdict({
        'New infections': ['new_infections'],
        'Cumulative infections': ['cum_infections'],
        'New reinfections': ['new_reinfections'],
    })
    if do_plot:
        sim.plot(do_save=do_save, do_show=do_show, fig_path=f'results/test_reinfection.png', to_plot=to_plot)

    return sim


def test_synthpops():
    sim = cv.Sim(pop_size=5000, pop_type='synthpops')
    sim.popdict = cv.make_synthpop(sim, with_facilities=True, layer_mapping={'LTCF': 'f'})
    sim.reset_layer_pars()

    # Vaccinate 75+, then 65+, then 50+, then 18+ on days 20, 40, 60, 80
    sim.vxsubtarg = sc.objdict()
    sim.vxsubtarg.age = [75, 65, 50, 18]
    sim.vxsubtarg.prob = [.05, .05, .05, .05]
    sim.vxsubtarg.days = subtarg_days = [20, 40, 60, 80]
    pfizer = cv.vaccinate(days=subtarg_days, vaccine_pars='pfizer', subtarget=vacc_subtarg)
    sim['interventions'] += [pfizer]

    sim.run()
    return sim



#%% Multisim and scenario tests

def test_vaccine_1strain_scen(do_plot=True, do_show=True, do_save=False):
    sc.heading('Run a basic sim with 1 strain, pfizer vaccine')

    sc.heading('Setting up...')

    # Define baseline parameters
    base_pars = {
        'n_days': 200,
    }

    n_runs = 3
    base_sim = cv.Sim(base_pars)

    # Vaccinate 75+, then 65+, then 50+, then 18+ on days 20, 40, 60, 80
    base_sim.vxsubtarg = sc.objdict()
    base_sim.vxsubtarg.age = [75, 65, 50, 18]
    base_sim.vxsubtarg.prob = [.05, .05, .05, .05]
    base_sim.vxsubtarg.days = subtarg_days = [20, 40, 60, 80]
    pfizer = cv.vaccinate(days=subtarg_days, vaccine_pars='pfizer', subtarget=vacc_subtarg)

    # Define the scenarios

    scenarios = {
        'baseline': {
            'name': 'No Vaccine',
            'pars': {}
        },
        'pfizer': {
            'name': 'Pfizer starting on day 20',
            'pars': {
                'interventions': [pfizer],
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
        scens.plot(do_save=do_save, do_show=do_show, fig_path=f'results/test_basic_vaccination.png', to_plot=to_plot)

    return scens


def test_vaccine_2strains_scen(do_plot=True, do_show=True, do_save=False):
    sc.heading('Run a basic sim with b117 strain on day 10, pfizer vaccine day 20')

    sc.heading('Setting up...')

    # Define baseline parameters
    base_pars = {
        'n_days': 250,
    }

    n_runs = 3
    base_sim = cv.Sim(base_pars)

    # Vaccinate 75+, then 65+, then 50+, then 18+ on days 20, 40, 60, 80
    base_sim.vxsubtarg = sc.objdict()
    base_sim.vxsubtarg.age = [75, 65, 50, 18]
    base_sim.vxsubtarg.prob = [.01, .01, .01, .01]
    base_sim.vxsubtarg.days = subtarg_days = [60, 150, 200, 220]
    jnj = cv.vaccinate(days=subtarg_days, vaccine_pars='j&j', subtarget=vacc_subtarg)
    b1351 = cv.Strain('b1351', days=10, n_imports=20)
    p1 = cv.Strain('p1', days=100, n_imports=100)

    # Define the scenarios

    scenarios = {
        'baseline': {
            'name': 'B1351 on day 10, No Vaccine',
            'pars': {
                'strains': [b1351]
            }
        },
        'b1351': {
            'name': 'B1351 on day 10, J&J starting on day 60',
            'pars': {
                'interventions': [jnj],
                'strains': [b1351],
            }
        },
        'p1': {
            'name': 'B1351 on day 10, J&J starting on day 60, p1 on day 100',
            'pars': {
                'interventions': [jnj],
                'strains': [b1351, p1],
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
        scens.plot(do_save=do_save, do_show=do_show, fig_path=f'results/test_vaccine_b1351.png', to_plot=to_plot)

    return scens


def test_strainduration_scen(do_plot=False, do_show=True, do_save=False):
    sc.heading('Run a sim with 2 strains, one of which has a much longer period before symptoms develop')
    sc.heading('Setting up...')

    strain_pars = {'dur':{'inf2sym': {'dist': 'lognormal_int', 'par1': 10.0, 'par2': 0.9}}}
    strains = cv.Strain(strain=strain_pars, strain_label='10 days til symptoms', days=10, n_imports=30)
    tp = cv.test_prob(symp_prob=0.2) # Add an efficient testing program

    base_pars = {
        'beta': 0.015, # Make beta higher than usual so people get infected quickly
        'n_days': 120,
        'interventions': tp
    }

    n_runs = 1
    base_sim = cv.Sim(base_pars)

    # Define the scenarios
    scenarios = {
        'baseline': {
            'name':'1 day to symptoms',
            'pars': {}
        },
        'slowsymp': {
            'name':'10 days to symptoms',
            'pars': {'strains': [strains]}
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


def test_msim():
    # basic test for vaccine
    b117 = cv.Strain('b117', days=0)
    sim = cv.Sim(strains=[b117])
    msim = cv.MultiSim(sim, n_runs=2)
    msim.run()
    msim.reduce()

    to_plot = sc.objdict({

        'Total infections': ['cum_infections'],
        'New infections per day': ['new_infections'],
        'New Re-infections per day': ['new_reinfections'],
        'New infections by strain': ['new_infections_by_strain']
    })

    msim.plot(to_plot=to_plot, do_save=0, do_show=1, legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=35)

    return msim


#%% Plotting and utilities

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


def vacc_subtarg(sim):
    ''' Subtarget by age'''

    # retrieves the first ind that is = or < sim.t
    ind = get_ind_of_min_value(sim.vxsubtarg.days, sim.t)
    age = sim.vxsubtarg.age[ind]
    prob = sim.vxsubtarg.prob[ind]
    inds = sc.findinds((sim.people.age>=age) * ~sim.people.vaccinated)
    vals = prob*np.ones(len(inds))
    return {'inds':inds, 'vals':vals}


def get_ind_of_min_value(list, time):
    ind = None
    for place, t in enumerate(list):
        if time >= t:
            ind = place

    if ind is None:
        errormsg = f'{time} is not within the list of times'
        raise ValueError(errormsg)
    return ind


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    # Run simplest possible test
    if 0:
         sim = cv.Sim()
         sim.run()

    # Run more complex single-sim tests
    sim0 = test_import1strain(do_plot=do_plot, do_save=do_save, do_show=do_show)
    # sim1 = test_import2strains(do_plot=do_plot, do_save=do_save, do_show=do_show)
    # sim2 = test_importstrain_longerdur(do_plot=do_plot, do_save=do_save, do_show=do_show)
    # sim3 = test_import2strains_changebeta(do_plot=do_plot, do_save=do_save, do_show=do_show)
    #
    # # Run Vaccine tests
    # sim4 = test_synthpops()
    # sim5 = test_vaccine_1strain()

    # # Run multisim and scenario tests
    # scens0 = test_vaccine_1strain_scen()
    # scens1 = test_vaccine_2strains_scen()
    # scens2 = test_strainduration_scen(do_plot=do_plot, do_save=do_save, do_show=do_show)
    # msim0 = test_msim()

    sc.toc()


print('Done.')

