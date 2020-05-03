'''
Compare simulating the entire population vs. dynamic rescaling vs. static rescaling.
'''

import sciris as sc
import covasim as cv

T = sc.tic()

p = sc.objdict() # Parameters
s = sc.objdict() # Sims
m = sc.objdict() # Multisims

# Interventions
iday = 60
cb = cv.change_beta(days=iday, changes=0.5) # Change beta
tn = cv.test_num(start_day=iday, daily_tests=1000, symp_test=10) # Test a number of people
tp = cv.test_prob(start_day=iday, symp_prob=0.1, asymp_prob=0.01) # Test a number of people

# Properties that are shared across sims
basepop      = 10e3
popscale     = 10
popinfected  = 20
which_interv = 2 # Which intervention to test

shared = sc.objdict(
    n_days = 120,
    beta = 0.015,
    interventions = [cb, tn, tp][which_interv],
)

# Simulate the entire population
p.entire = dict(
    pop_size     = basepop*popscale,
    pop_infected = popinfected,
    pop_scale    = 1,
    rescale      = False,
)

# Simulate a small population with dynamic scaling
p.rescale = dict(
    pop_size     = basepop,
    pop_infected = popinfected,
    pop_scale    = popscale,
    rescale      = True,
)

# Simulate a small population with static scaling
p.static = dict(
    pop_size     = basepop,
    pop_infected = popinfected//popscale,
    pop_scale    = popscale,
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

sc.toc(T)