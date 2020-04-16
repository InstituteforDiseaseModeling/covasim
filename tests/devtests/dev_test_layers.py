'''
Explore different layer options
'''

#%% Basic setup

import covasim as cv
import sciris as sc

do_plot = True

basepars = sc.objdict(
    pop_size     = 2000, # Population size
    pop_infected = 10,     # Number of initial infections
    n_days       = 90,   # Number of days to simulate
    rand_seed    = 1,     # Random seed
    verbose      = 0,
)


#%% Default
sc.heading('Default options')

pars = sc.objdict(
    pop_type     = 'random',
    use_layers   = False,
)

sim1 = cv.Sim(pars=sc.mergedicts(basepars, pars))
sim1.run(do_plot=do_plot)


#%% With layers
sc.heading('With layers options')

pars = sc.objdict(
    pop_type     = 'random',
    use_layers   = True,
)

sim2 = cv.Sim(pars=sc.mergedicts(basepars, pars))
sim2.run(do_plot=do_plot)

for label,sim in {'Default':sim1, 'With layers':sim2}.items():
    for key in sim1.people.contact_keys():
        print(f'{label}: layer {key} of length {len(sim.people.contacts[key])}')