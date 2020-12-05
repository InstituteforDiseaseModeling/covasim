'''
Full Covasim usage example, including a custom intervention
'''

import covasim as cv

def protect_elderly(sim):
    if sim.t == sim.day('2020-04-01'):
        elderly = sim.people.age>70
        sim.people.rel_sus[elderly] = 0.0

pars = dict(
    pop_size = 50e3,
    pop_infected = 100,
    n_days = 90,
    verbose = 0,
)

s1 = cv.Sim(pars, label='Default')
s2 = cv.Sim(pars, interventions=protect_elderly, label='Protect the elderly')
msim = cv.MultiSim([s1, s2])
msim.run(parallel=False) # NB, Jupyter notebooks can't run in parallel by default
fig = msim.plot(to_plot=['cum_deaths', 'cum_infections'])