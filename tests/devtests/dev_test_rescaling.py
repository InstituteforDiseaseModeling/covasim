'''
Compare simulating the entire population vs. dynamic rescaling vs. static rescaling.
'''

import sciris as sc
import covasim as cv

p = sc.objdict() # Parameters
s = sc.objdict() # Sims
m = sc.objdict() # Multisims

# Properties that are shared across sims
shared = sc.objdict(
    n_days = 120,
    beta = 0.012,
)

# Simulate the entire population
p.entire = dict(
    pop_size     = 500e3,
    pop_infected = 100,
    pop_scale    = 1,
    rescale      = False,
)

# Simulate a small population with dynamic scaling
p.rescale = dict(
    pop_size     = 25e3,
    pop_infected = 100,
    pop_scale    = 20,
    rescale      = True,
)

# Simulate a small population with static scaling
p.static = dict(
    pop_size     = 25e3,
    pop_infected = 5,
    pop_scale    = 20,
    rescale      = False,
)

keys = p.keys()

for key in keys:
    p[key].update(shared)

for key in keys:
    s[key] = cv.Sim(pars=p[key], label=key)
    m[key] = cv.MultiSim(base_sim=s[key], n_runs=10)

for key in keys:
    m[key].run()
    m[key].reduce()
    m[key].plot()

# msim = cv.MultiSim(sims=s.values(), reseed=False)
# msim.run()
# msim.compare()
# msim.plot()