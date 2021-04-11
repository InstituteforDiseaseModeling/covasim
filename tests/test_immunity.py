'''
Tests for immune waning, strains, and vaccine intervention.
'''

#%% Imports and settings
# import pytest
import sciris as sc
import covasim as cv
import numpy as np
import pandas as pd

do_plot = 0
do_save = 0
cv.options.set(interactive=False) # Assume not running interactively

base_pars = dict(
    pop_size = 1e3,
    verbose = -1,
)

#%% Define the tests

def test_waning(do_plot=False):
    sc.heading('Testing with and without waning')
    pars = dict(
        n_days = 500,
        beta = 0.008,
        NAb_decay = dict(form='nab_decay', pars={'init_decay_rate': 0.1, 'init_decay_time': 250, 'decay_decay_rate': 0.001})
    )
    s1 = cv.Sim(base_pars, **pars, use_waning=False, label='No waning').run()
    s2 = cv.Sim(base_pars, **pars, use_waning=True, label='With waning').run()
    msim = cv.MultiSim([s1,s2])
    if do_plot:
        msim.plot('overview-strain', rotation=30)
        sc.maximize()
    return msim


def test_states():
    # Load state diagram
    rawdf = pd.read_excel('state_diagram.xlsx', nrows=13)
    rawdf = rawdf.set_index('From ↓ to →')

    # Create and run simulation
    for use_waning in [False, True]:
        sc.heading(f'Testing state consistency with waning = {use_waning}')

        # Different states are possible with or without waning: resolve discrepancies
        df = sc.dcp(rawdf)
        if use_waning:
            df = df.replace(-0.5, 0)
            df = df.replace(-0.1, 1)
        else:
            df = df.replace(-0.5, -1)
            df = df.replace(-0.1, -1)

        pars = dict(
            pop_size = 1e3,
            pop_infected = 20,
            n_days = 70,
            use_waning = use_waning,
            verbose = 0,
            interventions = [
                cv.test_prob(symp_prob=0.4, asymp_prob=0.01),
                cv.contact_tracing(trace_probs=0.1),
            ]
        )
        sim = cv.Sim(pars).run()
        ppl = sim.people

        # Check states
        errormsg = ''
        states = df.columns.values.tolist()
        for s1 in states:
            for s2 in states:
                if s1 != s2:
                    relation = df.loc[s1, s2] # e.g. df.loc['susceptible', 'exposed']
                    print(f'Checking {s1:13s} → {s2:13s} = {relation:2n} ... ', end='')
                    inds     = cv.true(ppl[s1])
                    n_inds   = len(inds)
                    vals2    = ppl[s2][inds]
                    is_true  = cv.true(vals2)
                    is_false = cv.false(vals2)
                    n_true   = len(is_true)
                    n_false  = len(is_false)
                    if relation == 1 and n_true != n_inds:
                        errormsg = f'Being {s1}=True implies {s2}=True, but only {n_true}/{n_inds} people are'
                        print(f'× {n_true}/{n_inds} error!')
                    elif relation == -1 and n_false != n_inds:
                        errormsg = f'Being {s1}=True implies {s2}=False, but only {n_false}/{n_inds} people are'
                        print(f'× {n_true}/{n_inds} error!')
                    else:
                        print(f'✓ {n_true}/{n_inds}')
                    if errormsg:
                        raise RuntimeError(errormsg)

    return



# def test_varyingimmunity(do_plot=False, do_show=True, do_save=False):
#     sc.heading('Test varying properties of immunity')

#     # Define baseline parameters
#     n_runs = 3
#     base_sim = cv.Sim(use_waning=True, n_days=400, pars=base_pars)

#     # Define the scenarios
#     b1351 = cv.Strain('b1351', days=100, n_imports=20)

#     scenarios = {
#         'baseline': {
#             'name': 'Default Immunity (decay at log(2)/90)',
#             'pars': {
#                 'NAb_decay': dict(form='nab_decay', pars={'init_decay_rate': np.log(2)/90, 'init_decay_time': 250,
#                                                           'decay_decay_rate': 0.001}),
#             },
#         },
#         'faster_immunity': {
#             'name': 'Faster Immunity (decay at log(2)/30)',
#             'pars': {
#                 'NAb_decay': dict(form='nab_decay', pars={'init_decay_rate': np.log(2) / 30, 'init_decay_time': 250,
#                                                           'decay_decay_rate': 0.001}),
#             },
#         },
#         'baseline_b1351': {
#             'name': 'Default Immunity (decay at log(2)/90), B1351 on day 100',
#             'pars': {
#                 'NAb_decay': dict(form='nab_decay', pars={'init_decay_rate': np.log(2)/90, 'init_decay_time': 250,
#                                                           'decay_decay_rate': 0.001}),
#                 'strains': [b1351],
#             },
#         },
#         'faster_immunity_b1351': {
#             'name': 'Faster Immunity (decay at log(2)/30), B1351 on day 100',
#             'pars': {
#                 'NAb_decay': dict(form='nab_decay', pars={'init_decay_rate': np.log(2) / 30, 'init_decay_time': 250,
#                                                           'decay_decay_rate': 0.001}),
#                 'strains': [b1351],
#             },
#         },
#     }

#     metapars = {'n_runs': n_runs}
#     scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
#     scens.run()

#     to_plot = sc.objdict({
#         'New infections': ['new_infections'],
#         'New re-infections': ['new_reinfections'],
#         'Population Nabs': ['pop_nabs'],
#         'Population Immunity': ['pop_protection'],
#     })
#     if do_plot:
#         scens.plot(do_save=do_save, do_show=do_show, fig_path='results/test_basic_immunity.png', to_plot=to_plot)

#     return scens


# def test_import1strain(do_plot=False, do_show=True, do_save=False):
#     sc.heading('Test introducing a new strain partway through a sim')

#     strain_pars = {
#         'rel_beta': 1.5,
#     }
#     pars = {
#         'beta': 0.01
#     }
#     strain = cv.Strain(strain_pars, days=1, n_imports=20, label='Strain 2: 1.5x more transmissible')
#     sim = cv.Sim(use_waning=True, pars=pars, strains=strain, analyzers=cv.snapshot(30, 60), **pars, **base_pars)
#     sim.run()

#     return sim


# def test_import2strains(do_plot=False, do_show=True, do_save=False):
#     sc.heading('Test introducing 2 new strains partway through a sim')

#     b117 = cv.Strain('b117', days=1, n_imports=20)
#     p1 = cv.Strain('sa variant', days=2, n_imports=20)
#     sim = cv.Sim(use_waning=True, strains=[b117, p1], label='With imported infections', **base_pars)
#     sim.run()

#     return sim


# def test_importstrain_longerdur(do_plot=False, do_show=True, do_save=False):
#     sc.heading('Test introducing a new strain with longer duration partway through a sim')

#     pars = sc.mergedicts(base_pars, {
#         'n_days': 120,
#     })

#     strain_pars = {
#         'rel_beta': 1.5,
#         'dur': {'exp2inf':dict(dist='lognormal_int', par1=6.0,  par2=2.0)}
#     }

#     strain = cv.Strain(strain=strain_pars, label='Custom strain', days=10, n_imports=30)
#     sim = cv.Sim(use_waning=True, pars=pars, strains=strain, label='With imported infections')
#     sim.run()

#     return sim


# def test_import2strains_changebeta(do_plot=False, do_show=True, do_save=False):
#     sc.heading('Test introducing 2 new strains partway through a sim, with a change_beta intervention')

#     strain2 = {'rel_beta': 1.5,
#                'rel_severe_prob': 1.3}

#     strain3 = {'rel_beta': 2,
#                'rel_symp_prob': 1.6}

#     intervs  = cv.change_beta(days=[5, 20, 40], changes=[0.8, 0.7, 0.6])
#     strains  = [cv.Strain(strain=strain2, days=10, n_imports=20),
#                 cv.Strain(strain=strain3, days=30, n_imports=20),
#                ]
#     sim = cv.Sim(use_waning=True, interventions=intervs, strains=strains, label='With imported infections', **base_pars)
#     sim.run()

#     return sim



# #%% Vaccination tests

# def test_vaccine_1strain(do_plot=False, do_show=True, do_save=False):
#     sc.heading('Test vaccination with a single strain')
#     sc.heading('Setting up...')

#     pars = sc.mergedicts(base_pars, {
#         'beta': 0.015,
#         'n_days': 120,
#     })

#     pfizer = cv.vaccinate(days=[20], vaccine_pars='pfizer')
#     sim = cv.Sim(
#         use_waning=True,
#         pars=pars,
#         interventions=pfizer
#     )
#     sim.run()

#     to_plot = sc.objdict({
#         'New infections': ['new_infections'],
#         'Cumulative infections': ['cum_infections'],
#         'New reinfections': ['new_reinfections'],
#     })
#     if do_plot:
#         sim.plot(do_save=do_save, do_show=do_show, fig_path=f'results/test_reinfection.png', to_plot=to_plot)

#     return sim


# def test_synthpops():
#     sim = cv.Sim(use_waning=True, **sc.mergedicts(base_pars, dict(pop_size=5000, pop_type='synthpops')))
#     sim.popdict = cv.make_synthpop(sim, with_facilities=True, layer_mapping={'LTCF': 'f'})
#     sim.reset_layer_pars()

#     # Vaccinate 75+, then 65+, then 50+, then 18+ on days 20, 40, 60, 80
#     sim.vxsubtarg = sc.objdict()
#     sim.vxsubtarg.age = [75, 65, 50, 18]
#     sim.vxsubtarg.prob = [.05, .05, .05, .05]
#     sim.vxsubtarg.days = subtarg_days = [20, 40, 60, 80]
#     pfizer = cv.vaccinate(days=subtarg_days, vaccine_pars='pfizer', subtarget=vacc_subtarg)
#     sim['interventions'] += [pfizer]

#     sim.run()
#     return sim



# #%% Multisim and scenario tests

# def test_vaccine_1strain_scen(do_plot=False, do_show=True, do_save=False):
#     sc.heading('Run a basic sim with 1 strain, pfizer vaccine')

#     # Define baseline parameters
#     n_runs = 3
#     base_sim = cv.Sim(use_waning=True, pars=base_pars)

#     # Vaccinate 75+, then 65+, then 50+, then 18+ on days 20, 40, 60, 80
#     base_sim.vxsubtarg = sc.objdict()
#     base_sim.vxsubtarg.age = [75, 65, 50, 18]
#     base_sim.vxsubtarg.prob = [.05, .05, .05, .05]
#     base_sim.vxsubtarg.days = subtarg_days = [20, 40, 60, 80]
#     pfizer = cv.vaccinate(days=subtarg_days, vaccine_pars='pfizer', subtarget=vacc_subtarg)

#     # Define the scenarios

#     scenarios = {
#         'baseline': {
#             'name': 'No Vaccine',
#             'pars': {}
#         },
#         'pfizer': {
#             'name': 'Pfizer starting on day 20',
#             'pars': {
#                 'interventions': [pfizer],
#             }
#         },
#     }

#     metapars = {'n_runs': n_runs}
#     scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
#     scens.run()

#     to_plot = sc.objdict({
#         'New infections': ['new_infections'],
#         'Cumulative infections': ['cum_infections'],
#         'New reinfections': ['new_reinfections'],
#         # 'Cumulative reinfections': ['cum_reinfections'],
#     })
#     if do_plot:
#         scens.plot(do_save=do_save, do_show=do_show, fig_path='results/test_basic_vaccination.png', to_plot=to_plot)

#     return scens


# def test_vaccine_2strains_scen(do_plot=False, do_show=True, do_save=False):
#     sc.heading('Run a basic sim with b117 strain on day 10, pfizer vaccine day 20')

#     # Define baseline parameters
#     n_runs = 3
#     base_sim = cv.Sim(use_waning=True, pars=base_pars)

#     # Vaccinate 75+, then 65+, then 50+, then 18+ on days 20, 40, 60, 80
#     base_sim.vxsubtarg = sc.objdict()
#     base_sim.vxsubtarg.age = [75, 65, 50, 18]
#     base_sim.vxsubtarg.prob = [.01, .01, .01, .01]
#     base_sim.vxsubtarg.days = subtarg_days = [60, 150, 200, 220]
#     jnj = cv.vaccinate(days=subtarg_days, vaccine_pars='j&j', subtarget=vacc_subtarg)
#     b1351 = cv.Strain('b1351', days=10, n_imports=20)
#     p1 = cv.Strain('p1', days=100, n_imports=100)

#     # Define the scenarios

#     scenarios = {
#         'baseline': {
#             'name': 'B1351 on day 10, No Vaccine',
#             'pars': {
#                 'strains': [b1351]
#             }
#         },
#         'b1351': {
#             'name': 'B1351 on day 10, J&J starting on day 60',
#             'pars': {
#                 'interventions': [jnj],
#                 'strains': [b1351],
#             }
#         },
#         'p1': {
#             'name': 'B1351 on day 10, J&J starting on day 60, p1 on day 100',
#             'pars': {
#                 'interventions': [jnj],
#                 'strains': [b1351, p1],
#             }
#         },
#     }

#     metapars = {'n_runs': n_runs}
#     scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
#     scens.run()

#     to_plot = sc.objdict({
#         'New infections': ['new_infections'],
#         'Cumulative infections': ['cum_infections'],
#         'New reinfections': ['new_reinfections'],
#         # 'Cumulative reinfections': ['cum_reinfections'],
#     })
#     if do_plot:
#         scens.plot(do_save=do_save, do_show=do_show, fig_path='results/test_vaccine_b1351.png', to_plot=to_plot)

#     return scens


# def test_strainduration_scen(do_plot=False, do_show=True, do_save=False):
#     sc.heading('Run a sim with 2 strains, one of which has a much longer period before symptoms develop')

#     strain_pars = {'dur':{'inf2sym': {'dist': 'lognormal_int', 'par1': 10.0, 'par2': 0.9}}}
#     strains = cv.Strain(strain=strain_pars, label='10 days til symptoms', days=10, n_imports=30)
#     tp = cv.test_prob(symp_prob=0.2) # Add an efficient testing program

#     pars = sc.mergedicts(base_pars, {
#         'beta': 0.015, # Make beta higher than usual so people get infected quickly
#         'n_days': 120,
#         'interventions': tp
#     })
#     n_runs = 1
#     base_sim = cv.Sim(use_waning=True, pars=pars)

#     # Define the scenarios
#     scenarios = {
#         'baseline': {
#             'name':'1 day to symptoms',
#             'pars': {}
#         },
#         'slowsymp': {
#             'name':'10 days to symptoms',
#             'pars': {'strains': [strains]}
#         }
#     }

#     metapars = {'n_runs': n_runs}
#     scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
#     scens.run()

#     to_plot = sc.objdict({
#         'New infections': ['new_infections'],
#         'Cumulative infections': ['cum_infections'],
#         'New diagnoses': ['new_diagnoses'],
#         'Cumulative diagnoses': ['cum_diagnoses'],
#     })
#     if do_plot:
#         scens.plot(do_save=do_save, do_show=do_show, fig_path='results/test_strainduration.png', to_plot=to_plot)

#     return scens


# def test_waning_vs_not(do_plot=False, do_show=True, do_save=False):
#     sc.heading('Testing waning...')

#     # Define baseline parameters
#     pars = sc.mergedicts(base_pars, {
#         'pop_size': 10e3,
#         'pop_scale': 50,
#         'n_days': 150,
#         'use_waning': False,
#     })

#     n_runs = 3
#     base_sim = cv.Sim(pars=pars)

#     # Define the scenarios
#     scenarios = {
#         'no_waning': {
#             'name': 'No waning',
#             'pars': {
#             }
#         },
#         'waning': {
#             'name': 'Waning',
#             'pars': {
#                 'use_waning': True,
#             }
#         },
#     }

#     metapars = {'n_runs': n_runs}
#     scens = cv.Scenarios(sim=base_sim, metapars=metapars, scenarios=scenarios)
#     scens.run()

#     to_plot = sc.objdict({
#         'New infections': ['new_infections'],
#         'New reinfections': ['new_reinfections'],
#         'Cumulative infections': ['cum_infections'],
#         'Cumulative reinfections': ['cum_reinfections'],
#     })
#     if do_plot:
#         scens.plot(do_save=do_save, do_show=do_show, fig_path='results/test_waning_vs_not.png', to_plot=to_plot)

#     return scens


# def test_msim(do_plot=False):
#     sc.heading('Testing multisim...')

#     # basic test for vaccine
#     b117 = cv.Strain('b117', days=0)
#     sim = cv.Sim(use_waning=True, strains=[b117], **base_pars)
#     msim = cv.MultiSim(sim, n_runs=2)
#     msim.run()
#     msim.reduce()

#     to_plot = sc.objdict({
#         'Total infections': ['cum_infections'],
#         'New infections per day': ['new_infections'],
#         'New Re-infections per day': ['new_reinfections'],
#     })

#     if do_plot:
#         msim.plot(to_plot=to_plot, do_save=0, do_show=1, legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=35)

#     return msim


#%% Plotting and utilities

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

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    # msim1 = test_waning(do_plot=do_plot)
    sim1  = test_states()

    sc.toc(T)
    print('Done.')
