'''
Example script to vary relative transmissibility and susceptibility for the elderly
'''

#%% Imports and settings
import covasim as cv
import copy

sim = cv.Sim()
sim.initialize()

new_prognoses = copy.deepcopy(sim.pars['prognoses'])
trans_ORs     = new_prognoses['trans_ORs']
sus_ORs       = new_prognoses['sus_ORs']
age_cutoffs   = new_prognoses['age_cutoffs']

trans_ORs[age_cutoffs>=70]  *= 0.8 # Reduce relative transmissibility for people over 70 by 20%
sus_ORs[age_cutoffs>=70]    *= 0.8 # Reduce relative susceptibility for people over 70 by 20%

n_people = sim['pop_size']
n_days   = sim['n_days']

# Define the scenarios
scenarios = {
    'baseline': {
        'name': 'Baseline',
        'pars': {
        }
    },
    'protectelderly': {
        'name': 'Protect the elderly',
        'pars': {'prognoses': new_prognoses,
                 'interventions': cv.test_num(daily_tests=[0.10*n_people]*n_days, subtarget={'inds': sim.people.age>50, 'val': 1.2})}
    },
}

to_plot = [
    'cum_infections',
    'new_infections',
    'n_severe',
    'cum_deaths',
]
fig_args = dict(figsize=(24, 16))

scens = cv.Scenarios(sim=sim, scenarios=scenarios, metapars={'n_runs': 1})
scens.run()
scens.plot()



