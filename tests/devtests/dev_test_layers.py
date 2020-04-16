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
sc.heading('Default options')

pars = sc.objdict(
    pop_type     = 'random',
    use_layers   = True,
)

sim2 = cv.Sim(pars=sc.mergedicts(basepars, pars))
sim2.run(do_plot=do_plot)