'''
Simple script for running the Covid-19 agent-based model
'''

# import sciris as sc
# import covasim as cv
# from guppy import hpy

# pars = sc.objdict(
#     pop_size     = 20000, # Population size
#     pop_infected = 1,     # Number of initial infections
#     n_days       = 60,   # Number of days to simulate
#     rand_seed    = 1,     # Random seed
#     pop_type     = 'hybrid',
#     verbose      = 0,
# )

# sim = cv.Sim(pars=pars)
# sim.run()

# h = hpy()
# print(h.heap())

# Benchmark the simulation

import sciris as sc
import covasim as cv

def make_run_sim():
    sim = cv.Sim(n_days=180, verbose=0)
    sim.init_people()
    sim.initialize()
    sim.run()
    del sim.people.contacts
    del sim.people
    del sim.popdict
    return sim

sim = make_run_sim()

to_profile = 'run' # Must be one of the options listed below...currently only 1

sc.mprofile(run=make_run_sim, follow=make_run_sim)
