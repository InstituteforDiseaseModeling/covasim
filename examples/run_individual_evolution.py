'''
Simple script for running Covasim
'''

import sciris as sc
import covasim as cv
import numpy as np
import pandas as pd

# Run options
do_plot = 1
verbose = 1
interv  = 1

# Configure the sim -- can also just use a normal dictionary
pars = sc.objdict(
    pop_size     = 600,    # Population size
    pop_infected = 10,       # Number of initial infections
    n_days       = 120,      # Number of days to simulate
    rand_seed    = 1,        # Random seed
    pop_type     = 'hybrid', # Population to use -- "hybrid" is random with household, school,and work structure
    dynam_layer  = dict(h=False,   s=False,   w=True,   c=False),
)

# Optionally add an intervention
if interv:
    pars.interventions = cv.change_beta(days=45, changes=0.5)

# Make, run, and plot the sim
sim = cv.Sim(pars=pars)
sim.initialize()
results = sim.run(verbose=verbose)
for k1 in ["contacts", "state"]:
    for k2 in results["evol"][k1].keys():
        results["evol"][k1][k2] = np.array(results["evol"][k1][k2])

pd.DataFrame.from_dict(results["evol"]).to_pickle("evolution.pandas_pickle")
np.savez("static_contacts.npz", results["static_contacts"])
if do_plot:
    sim.plot()
