'''
Compare simulating the entire population vs. dynamic rescaling vs. static rescaling.
'''

import sciris as sc
import covasim as cv

p = sc.objdict() # Parameters
s = sc.objdict() # Sims
m = sc.objdict() # Multisims

# Interventions
cb = cv.change_beta(days=60, changes=0.5) # Change beta
# tn = cv.test_num() # Test a number of people
# tp = cv.test_prob() # Test a number of people


# Properties that are shared across sims
shared = sc.objdict(
    n_days = 120,
    beta = 0.012,
    interventions = cb,
)

# Simulate the entire population
p.entire = dict(
    pop_size     = 200e3,
    pop_infected = 20,
    pop_scale    = 1,
    rescale      = False,
)

# Simulate a small population with dynamic scaling
p.rescale = dict(
    pop_size     = 20e3,
    pop_infected = 20,
    pop_scale    = 10,
    rescale      = True,
)

# Simulate a small population with static scaling
p.static = dict(
    pop_size     = 20e3,
    pop_infected = 2,
    pop_scale    = 10,
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

bsims = [msim.base_sim for msim in m.values()]
mm = cv.MultiSim(sims=bsims)
mm.compare()

# msim = cv.MultiSim(sims=s.values(), reseed=False)
# msim.run()
# msim.compare()
# msim.plot()