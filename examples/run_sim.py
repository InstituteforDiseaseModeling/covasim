'''
Simple script for running the Covid-19 agent-based model
'''

print('Importing...')
import sciris as sc
import covasim as cv
import fire

# Set filename if saving
version  = 'v0'
date     = '2020apr06'
folder   = 'results'
basename = f'{folder}/covasim_run_{date}_{version}'
fig_path = f'{basename}.png'

# Configure the sim -- can also just use a normal dictionary
pars = sc.objdict(
    pop_size     = 20000, # Population size
    pop_infected = 1,     # Number of initial infections
    n_days       = 180,   # Number of days to simulate
    rand_seed    = 1,     # Random seed
    )

def run(pars=pars, 
        interv=False,
        verbose=True,
        do_plot=True, 
        do_save=False, 
        do_show=True,
        fig_path=fig_path
        ):

    '''
    Run simulation, optionally with an intervention.

    To run simulation with default parameters:
    > python rum_sim.py

    To run simulation with an intervention:
    > python run_sim.py --interv=True

    To run simulation with an intervention and save plot to disk:
    > python run_sim.py --interv=True --do_save=True

    To run simulation with custom configuration
    > python run_sim.py --pars "{pop_size:20000, pop_infected:1, n_days:360, rand_seed:1}"

    Args:
        pars: (dict): configuration for the sim.  See description for example.
        interv: (bool): whether or not to add an intervention, specified by cv.change_beta(days=45, changes=0.5)
        do_plot: (bool): whether or not to generate a plot.  Defaults to True.
        do_save: (bool): If a plot is generated, whether or not to save it.  Defaults to False.
        do_show: (bool): If a plot is generated, whether or not to show it.  Defaults to True.
        fig_path: (str): Path to which save filename.  Defaults to results/covasim_run_{date}_{version}.png
        verbose: (bool): whether or not turn verbose mode while running simulation
    '''
    if interv: 
        pars.interventions = cv.change_beta(days=45, changes=0.5) # Optionally add an intervention
    
    print('Making sim...')
    sim = cv.Sim(pars=pars)

    print('Running...')
    sim.run(verbose=verbose)

    if do_plot:
        print('Plotting...')
        fig = sim.plot(do_save=do_save, do_show=do_show, fig_path=fig_path)

if __name__ == '__main__':
    fire.Fire(run)
