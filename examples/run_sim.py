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
date     = '2020apr15'
folder   = 'results'
basename = f'{folder}/covasim_run_{date}_{version}'
fig_path = f'{basename}.png'

# Configure the sim -- can also just use a normal dictionary
pars = sc.objdict(
    pop_size     = 20000, # Population size
    pop_infected = 1,     # Number of initial infections
    n_days       = 60,   # Number of days to simulate
    rand_seed    = 1,     # Random seed
    pop_type     = 'random',
    use_layers   = True,
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
