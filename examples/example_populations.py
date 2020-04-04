import covasim as cv
import sciris as sc

# Random network using defaults
sim = cv.Sim()
sim.run()
fig = sim.plot()

# Random network with specified beta and people
sim = cv.Sim({'beta':0.01, 'n':2000})
sim.run()
fig = sim.plot()

# Make and use a random population (including community transmission)
pars = cv.make_pars()
pars['population'] = cv.Population.random(pars, n_random_contacts=20)
sim = cv.Sim(pars=pars)
sim.run()
fig = sim.plot()

# Add an intervention changing the overall beta
pars['interventions'] = cv.interventions.change_beta(25, 0.5)
sim = cv.Sim(pars=pars)
sim.run()
fig = sim.plot()

# Change beta only for community contacts
pars['interventions'] = cv.interventions.change_beta(25, 0.0, pars['population'].contact_layers['Community'])
sim = cv.Sim(pars=pars)
sim.run()
fig = sim.plot()

# Run sim using synthpops with community transmission
pars = cv.make_pars()
pars['population'] = cv.Population.synthpops(pars, n_people=5000, n_random_contacts=20)
sim = cv.Sim(pars=pars)
sim.run()
fig = sim.plot()
