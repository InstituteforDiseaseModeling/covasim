'''
Compare simulating the entire population vs. dynamic rescaling vs. static rescaling.
'''

import sciris as sc
import covasim as cv

p = sc.objdict() # Parameters
s = sc.objdict() # Sims
m = sc.objdict() # Multisims

shared = sc.objdict(
    n_days = 120,
    beta = 0.012,
    )


p.entire = dict(
    pop_size = 200e3,
    pop_infected = 100,
    pop_scale = 1,
    rescale = False,
)

p.rescale = dict(
    pop_size = 20e3,
    pop_infected = 100,
    pop_scale = 10,
    rescale = True,
)

p.static = dict(
    pop_size = 20e3,
    pop_infected = 10,
    pop_scale = 10,
    rescale = False,
)

keys = p.keys()

for key in keys:
    p[key].update(shared)

for key in keys:
    s[key] = cv.Sim(pars=p[key], label=key)

msim = cv.MultiSim(sims=s.values(), reseed=False)
msim.run()
msim.compare()
msim.plot()