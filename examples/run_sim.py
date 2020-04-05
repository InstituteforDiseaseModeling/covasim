'''
Simple script for running the Covid-19 agent-based model
'''

print('Importing...')
import sciris as sc
import covasim as cv

print('Configuring...')

# Run options
do_plot = 1
do_save = 0
do_show = 1
verbose = 1
interv  = 0

# Set filename if saving
version  = 'v0'
date     = '2020apr05'
folder   = 'results'
basename = f'{folder}/covasim_run_{date}_{version}'
fig_path = f'{basename}.png'

# Configure the sim -- can also just use a normal dictionary
pars = sc.objdict(
    n           = 20000, # Population size
    n_infected  = 1,    # Number of initial infections
    n_days      = 180,   # Number of days to simulate
    prog_by_age = 1,    # Use age-specific mortality etc.
    usepopdata  = 1,    # Use realistic population structure (requires synthpops)
    seed        = 1,    # Random seed
    )

# Optionally add an intervention
if interv:
    pars.interventions = cv.change_beta(days=45, changes=0.5) # Optionally add an intervention

print('Making sim...')
sim = cv.Sim(pars=pars)

print('Running...')
sim.run(verbose=verbose)

if do_plot:
    print('Plotting...')
    fig = sim.plot(do_save=do_save, do_show=do_show, fig_path=fig_path)










