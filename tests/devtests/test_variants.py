import covasim as cv
import sciris as sc
import numpy as np


do_plot = 0
do_show = 0
do_save = 0
debug   = 0

base_pars = dict(
    pop_size = 10e3,
    verbose = -1,
)

def test_simple(do_plot=False):
    s1 = cv.Sim(base_pars).run()
    s2 = cv.Sim(base_pars, n_days=300, use_waning=True).run()
    if do_plot:
        s1.plot()
        s2.plot()
    return


def test_import1strain(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test introducing a new strain partway through a sim')

    strain_pars = {
        'rel_beta': 1.5,
    }
    pars = {
        'beta': 0.01
    }
    strain = cv.strain(strain_pars, days=1, n_imports=20, label='Strain 2: 1.5x more transmissible')
    sim = cv.Sim(use_waning=True, pars=pars, strains=strain, analyzers=cv.snapshot(30, 60), **pars, **base_pars)
    sim.run()

    return sim


def test_import2strains(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test introducing 2 new strains partway through a sim')

    b117 = cv.strain('b117', days=1, n_imports=20)
    p1 = cv.strain('sa variant', days=2, n_imports=20)
    sim = cv.Sim(use_waning=True, strains=[b117, p1], label='With imported infections', **base_pars)
    sim.run()

    return sim


def test_importstrain_longerdur(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test introducing a new strain with longer duration partway through a sim')

    pars = sc.mergedicts(base_pars, {
        'n_days': 120,
    })

    strain_pars = {
        'rel_beta': 1.5,
    }

    strain = cv.strain(strain=strain_pars, label='Custom strain', days=10, n_imports=30)
    sim = cv.Sim(use_waning=True, pars=pars, strains=strain, label='With imported infections')
    sim.run()

    return sim


def test_import2strains_changebeta(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test introducing 2 new strains partway through a sim, with a change_beta intervention')

    strain2 = {'rel_beta': 1.5,
               'rel_severe_prob': 1.3}

    strain3 = {'rel_beta': 2,
               'rel_symp_prob': 1.6}

    intervs  = cv.change_beta(days=[5, 20, 40], changes=[0.8, 0.7, 0.6])
    strains  = [cv.strain(strain=strain2, days=10, n_imports=20),
                cv.strain(strain=strain3, days=30, n_imports=20),
               ]
    sim = cv.Sim(use_waning=True, interventions=intervs, strains=strains, label='With imported infections', **base_pars)
    sim.run()

    return sim



#%% Vaccination tests

def test_vaccine_1strain(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test vaccination with a single strain')

    pars = sc.mergedicts(base_pars, {
        'beta': 0.015,
        'n_days': 120,
    })

    pfizer = cv.vaccinate(days=[20], vaccine='pfizer')
    sim = cv.Sim(
        use_waning=True,
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
        sim.plot(do_save=do_save, do_show=do_show, fig_path='results/test_reinfection.png', to_plot=to_plot)

    return sim


def test_synthpops():
    sim = cv.Sim(use_waning=True, **sc.mergedicts(base_pars, dict(pop_size=5000, pop_type='synthpops')))
    sim.popdict = cv.make_synthpop(sim, with_facilities=True, layer_mapping={'LTCF': 'f'})
    sim.reset_layer_pars()

    # Vaccinate 75+, then 65+, then 50+, then 18+ on days 20, 40, 60, 80
    sim.vxsubtarg = sc.objdict()
    sim.vxsubtarg.age = [75, 65, 50, 18]
    sim.vxsubtarg.prob = [.05, .05, .05, .05]
    sim.vxsubtarg.days = subtarg_days = [20, 40, 60, 80]
    pfizer = cv.vaccinate(days=subtarg_days, vaccine='pfizer', subtarget=vacc_subtarg)
    sim['interventions'] += [pfizer]

    sim.run()
    return sim



#%% Multisim and scenario tests

def test_vaccine_1strain_scen(do_plot=False, do_show=True, do_save=False):
    sc.heading('Run a basic sim with 1 strain, pfizer vaccine')

    # Define baseline parameters
    n_runs = 3
    base_sim = cv.Sim(use_waning=True, pars=base_pars)

    # Vaccinate 75+, then 65+, then 50+, then 18+ on days 20, 40, 60, 80
    base_sim.vxsubtarg = sc.objdict()
    base_sim.vxsubtarg.age = [75, 65, 50, 18]
    base_sim.vxsubtarg.prob = [.05, .05, .05, .05]
    base_sim.vxsubtarg.days = subtarg_days = [20, 40, 60, 80]
    pfizer = cv.vaccinate(days=subtarg_days, vaccine='pfizer', subtarget=vacc_subtarg)

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
        scens.plot(do_save=do_save, do_show=do_show, fig_path='results/test_basic_vaccination.png', to_plot=to_plot)

    return scens


def test_vaccine_2strains_scen(do_plot=False, do_show=True, do_save=False):
    sc.heading('Run a basic sim with b117 strain on day 10, pfizer vaccine day 20')

    # Define baseline parameters
    n_runs = 3
    base_sim = cv.Sim(use_waning=True, pars=base_pars)

    # Vaccinate 75+, then 65+, then 50+, then 18+ on days 20, 40, 60, 80
    base_sim.vxsubtarg = sc.objdict()
    base_sim.vxsubtarg.age = [75, 65, 50, 18]
    base_sim.vxsubtarg.prob = [.01, .01, .01, .01]
    base_sim.vxsubtarg.days = subtarg_days = [60, 150, 200, 220]
    jnj = cv.vaccinate(days=subtarg_days, vaccine='j&j', subtarget=vacc_subtarg)
    b1351 = cv.strain('b1351', days=10, n_imports=20)
    p1 = cv.strain('p1', days=100, n_imports=100)

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
    scens.run(debug=debug)

    to_plot = sc.objdict({
        'New infections': ['new_infections'],
        'Cumulative infections': ['cum_infections'],
        'New reinfections': ['new_reinfections'],
        # 'Cumulative reinfections': ['cum_reinfections'],
    })
    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path='results/test_vaccine_b1351.png', to_plot=to_plot)

    return scens


def test_msim(do_plot=False):
    sc.heading('Testing multisim...')

    # basic test for vaccine
    b117 = cv.strain('b117', days=0)
    sim = cv.Sim(use_waning=True, strains=[b117], **base_pars)
    msim = cv.MultiSim(sim, n_runs=2)
    msim.run()
    msim.reduce()

    to_plot = sc.objdict({
        'Total infections': ['cum_infections'],
        'New infections per day': ['new_infections'],
        'New Re-infections per day': ['new_reinfections'],
    })

    if do_plot:
        msim.plot(to_plot=to_plot, do_save=0, do_show=1, legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=35)

    return msim


def test_varyingimmunity(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test varying properties of immunity')

    # Define baseline parameters
    n_runs = 3
    base_sim = cv.Sim(use_waning=True, n_days=400, pars=base_pars)

    # Define the scenarios
    b1351 = cv.strain('b1351', days=100, n_imports=20)

    scenarios = {
        'baseline': {
            'name': 'Default Immunity (decay at log(2)/90)',
            'pars': {
                'nab_decay': dict(form='nab_decay', decay_rate1=np.log(2)/90, decay_time1=250, decay_rate2=0.001),
            },
        },
        'faster_immunity': {
            'name': 'Faster Immunity (decay at log(2)/30)',
            'pars': {
                'nab_decay': dict(form='nab_decay', decay_rate1=np.log(2)/30, decay_time1=250, decay_rate2=0.001),
            },
        },
        'baseline_b1351': {
            'name': 'Default Immunity (decay at log(2)/90), B1351 on day 100',
            'pars': {
                'nab_decay': dict(form='nab_decay', decay_rate1=np.log(2)/90, decay_time1=250, decay_rate2=0.001),
                'strains': [b1351],
            },
        },
        'faster_immunity_b1351': {
            'name': 'Faster Immunity (decay at log(2)/30), B1351 on day 100',
            'pars': {
                'nab_decay': dict(form='nab_decay', decay_rate1=np.log(2)/30, decay_time1=250, decay_rate2=0.001),
                'strains': [b1351],
            },
        },
    }

    metapars = {'n_runs': n_runs}
    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run(debug=debug)

    to_plot = sc.objdict({
        'New infections': ['new_infections'],
        'New re-infections': ['new_reinfections'],
        'Population Nabs': ['pop_nabs'],
        'Population Immunity': ['pop_protection'],
    })
    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path='results/test_basic_immunity.png', to_plot=to_plot)

    return scens


def test_waning_vs_not(do_plot=False, do_show=True, do_save=False):
    sc.heading('Testing waning...')

    # Define baseline parameters
    pars = sc.mergedicts(base_pars, {
        'pop_size': 10e3,
        'pop_scale': 50,
        'n_days': 150,
        'use_waning': False,
    })

    n_runs = 3
    base_sim = cv.Sim(pars=pars)

    # Define the scenarios
    scenarios = {
        'no_waning': {
            'name': 'No waning',
            'pars': {
            }
        },
        'waning': {
            'name': 'Waning',
            'pars': {
                'use_waning': True,
            }
        },
    }

    metapars = {'n_runs': n_runs}
    scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
    scens.run()

    to_plot = sc.objdict({
        'New infections': ['new_infections'],
        'New reinfections': ['new_reinfections'],
        'Cumulative infections': ['cum_infections'],
        'Cumulative reinfections': ['cum_reinfections'],
    })
    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path='results/test_waning_vs_not.png', to_plot=to_plot)

    return scens


#%% Utilities

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

    # Gather keywords
    kw = dict(do_plot=do_plot, do_save=do_save, do_show=do_show)

    # Run simplest possible test
    test_simple(do_plot=do_plot)

    # Run more complex single-sim tests
    sim0 = test_import1strain(**kw)
    sim1 = test_import2strains(**kw)
    sim2 = test_importstrain_longerdur(**kw)
    sim3 = test_import2strains_changebeta(**kw)

    # Run Vaccine tests
    sim4 = test_synthpops()
    sim5 = test_vaccine_1strain()

    # Run multisim and scenario tests
    scens0 = test_vaccine_1strain_scen()
    scens1 = test_vaccine_2strains_scen()
    msim0  = test_msim()

    # Run immunity tests
    sim_immunity0 = test_varyingimmunity(**kw)

    # Run test to compare sims with and without waning
    scens2 = test_waning_vs_not(**kw)

    sc.toc()


print('Done.')

