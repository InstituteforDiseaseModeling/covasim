'''
Set the parameters for LEMOD-FP.
'''

import os
import pylab as pl
import sciris as sc


#%% Set parameters for the simulation

def make_pars():
    pars = {}

    # Simulation parameters
    pars['name'] = 'Default' # Name of the simulation
    pars['n'] = 3720 # Number of people in the simulation
    pars['start'] = 0 # Start day of the epidemic
    pars['end'] = 30 # How many days to simulate
    pars['timestep'] = 1 # Timestep in days
    pars['verbose'] = True
    pars['seed'] = 1 # Random seed, if None, don't reset
    
    # User-tunable parameters
    pars['mortality_factor']    = 1.0#*(2**2) # These weird factors are since mortality and fertility scale differently to keep population growth the same
    
    return pars