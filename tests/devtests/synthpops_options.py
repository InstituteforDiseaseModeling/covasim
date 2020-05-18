import covasim as cv


sim = cv.Sim(pop_size=5000, pop_type='synthpops')
popdict, layer_keys = cv.make_synthpop(sim, ltcf=False, layer_mapping={'LTCF':'ltcf'})
sim.popdict = popdict
sim.initialize()
sim.run()
