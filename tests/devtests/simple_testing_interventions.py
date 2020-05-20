'''
Perform simple testing tests
'''

import covasim as cv

pars = dict(
    n_days = 180,
    )

sim = cv.Sim(pars)
sim.initialize()
n_people = sim['pop_size']
n_tests  = 0.1 * n_people

delay = 5
start_day = 40
scenarios = {
    'test_num': {
        'name': 'test_num',
        'pars': {
            'interventions': cv.test_num(daily_tests=n_tests, symp_test=1.0, start_day=start_day, test_delay=delay)
            }
    },
    'test_prob': {
        'name': 'test_prob',
        'pars': {
            'interventions': cv.test_prob(symp_prob=0.1, asymp_prob=0.1, start_day=start_day, test_delay=delay)
        }
    },
}

scens = cv.Scenarios(sim=sim, scenarios=scenarios, metapars={'n_runs': 1})
scens.run(debug=True)
scens.plot(to_plot='overview', n_cols=3)