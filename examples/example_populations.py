import covasim as cv

default_n = 1000 # Default number of people

# Random network using defaults
sim = cv.Sim({'n':default_n})
sim.run()
fig = sim.plot()


# Random network with specified beta and people
sim = cv.Sim({'beta':0.01, 'n':2000})
sim.run()
fig = sim.plot()

# Default is to use age-specific progression. Here we modify the parameters to remove
# age variation
pars = cv.make_pars(n=default_n)
pars['prognoses'] = cv.get_default_prognoses(by_age=False) # Replace the prognoses with the non age specific default values
sim = cv.Sim()
sim.run()
fig = sim.plot()

# Make and use a random population (including community transmission)
pars = cv.make_pars(n=default_n)
population = cv.Population.random(pars, n_random_contacts=20)
sim = cv.Sim(pars=pars, population=population)
sim.run()
fig = sim.plot()

# Add an intervention changing the overall beta
pars = cv.make_pars(n=default_n)
population = cv.Population.random(pars, n_random_contacts=20)
pars['interventions'] = cv.interventions.change_beta(25, 0.5)
sim = cv.Sim(pars=pars, population=population)
sim.run()
fig = sim.plot()

# Change beta only for community contacts
pars = cv.make_pars(n=default_n)
population = cv.Population.random(pars, n_random_contacts=20)
pars['interventions'] = cv.interventions.change_beta(25, 0.0, population.contact_layers['Community'])
sim = cv.Sim(pars=pars, population=population)
sim.run()
fig = sim.plot()

# Run sim using synthpops with community transmission
if cv.requirements.available['synthpops']:
    pars = cv.make_pars()
    population = cv.Population.synthpops(pars, n_people=5000, n_random_contacts=20)
    sim = cv.Sim(pars=pars, population=population)
    sim.run()
    fig = sim.plot()
