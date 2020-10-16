import covasim as cv
from SimVorarlberg.pars import pars
from SimVorarlberg.specialInterventions.testIntervention import change_beta_by_age

# Run options
do_plot = 1
do_show = 1
verbose = 1

# Scenario metaparameters
metapars = dict(
    n_runs    = 3, # Number of parallel runs; change to 3 for quick, 11 for real
    noise     = 0.1, # Use noise, optionally
    noisepar  = 'beta',
    rand_seed = 1,
    quantiles = {'low':0.4, 'high':0.6},
)

intervention_start_day = 60


if __name__ == '__main__':

    scenarios = {
        'baseline': {
            'name': 'Baseline',
            'pars': {
                'interventions': None,
            }
        },
        'sd_al': {
            'name': 'Social distancing all layers',
            'pars': {
                'interventions': cv.change_beta(days=intervention_start_day, changes=0.7, layers=['s','w','c', 'h']),
            }
        },
        'sd_wsc': {
            'name': 'Social distancing layers; work, school, community',
            'pars': {
                'interventions': cv.change_beta(days=intervention_start_day, changes=0.7, layers=['s','w','c'])
            }
        },
        'cs_ce': {
            'name': 'closing schools by cutting edges',
            'pars': {
                'interventions': cv.clip_edges(days=intervention_start_day, changes=0, layers=['s'])
            }
        },
        'cs_b0': {
            'name': 'closing schools by setting beta to 0',
            'pars': {
                'interventions': cv.change_beta(days=intervention_start_day, changes=0, layers=['s'])
            }
        },
        'cb_55': {
            'name': 'change beta for persons of age 55 an higher',
            'pars': {
                'interventions': change_beta_by_age(days=intervention_start_day, changes=0, age=55)
            }
        },
    }

    sim = cv.Sim(pars)
    sim.init_people(load_pop=True, popfile='testPop.pop')

    scens = cv.Scenarios(sim=sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose)
    if do_plot:
        fig1 = scens.plot(do_show=do_show)



