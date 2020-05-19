'''
Script for testing integration with new SynthPops features -- namely, long-term
care facilities. Not currently functional.
'''

import covasim as cv

pars = dict(
    pop_size=20000,
    pop_type='synthpops',
    n_days=120,
    )

sims = []
for ltcf in [False, True]:
    print(f'Running LTCF {ltcf}')
    sim = cv.Sim(pars)
    popdict, layer_keys = cv.make_synthpop(sim, with_facilities=ltcf, layer_mapping={'LTCF':'f'})
    sim.popdict = popdict
    sim.reset_layer_pars(layer_keys)
    sims.append(sim)

to_plot = ['cum_infections', 'new_infections', 'cum_severe', 'cum_deaths']
msim = cv.MultiSim(sims)
msim.run()
msim.plot(to_plot=to_plot)