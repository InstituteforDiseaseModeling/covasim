'''
Check if states match
'''

import sciris as sc
import covasim as cv

states = [
        'susceptible',
        'exposed',
        'infectious',
        'symptomatic',
        'severe',
        'critical',
        'tested',
        'diagnosed',
        'recovered',
        'dead',
]

sim = cv.Sim()
sim.run()

d = sc.objdict()
for state in states:
    n_in = len(list(sim.people.filter_in(state)))
    n_out = len(list(sim.people.filter_out(state)))
    d[state] = n_in
    assert n_in + n_out == sim['pop_size']

print(sim.summary)
print(d)
assert d.susceptible + d.exposed + d.recovered + d.dead == sim['pop_size']
