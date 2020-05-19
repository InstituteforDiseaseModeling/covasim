'''
Script for testing integration with new SynthPops features -- namely, long-term
care facilities. Not currently functional.
'''

import covasim as cv

sim = cv.Sim(pop_size=5000, pop_type='synthpops')
popdict, layer_keys = cv.make_synthpop(sim, with_facilities=False, layer_mapping={'LTCF':'f'})
sim.popdict = popdict
sim.initialize()
sim.run()
