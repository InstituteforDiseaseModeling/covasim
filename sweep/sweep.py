import os, pickle
import sciris as sc
import covasim as cv
import fire
import wandb
from pathlib import Path

def train(beta:float=0.015, 
          pop_infected: int=10, 
          rel_death_prob: float=1.0,
          rel_severe_prob: float=1.0,
          rel_crit_prob: float=1.0,
          start_day: str='2019-12-25',
          datafile='tests/example_data.csv') -> None:
    """
    Perform hyperparameter sweep with Weights and Biases
    https://docs.wandb.com/sweeps
    """
    assert Path(datafile).exists(), f'Not able to find file: {datafile}'

    pars = dict(
        beta = beta,
        pop_infected = pop_infected,
        rel_death_prob = rel_death_prob,
        rel_crit_prob = rel_crit_prob,
        start_day = start_day,
        )

    # instantiate wandb run
    wb_handle = wandb.init(config=pars, project="covasim")
    run_id = wandb.run.id

    # Create and run the simulation
    sc.heading('Hyperparmeter Sweep')
    sim = cv.Sim(pars=pars, datafile=datafile)
    sim.run(verbose=False)
    likelihood = sim.likelihood()

    # log relevant metrics and artifacts
    wandb.log({'likelihood': likelihood})
    sim.plot(do_show=False, 
             do_save=True, 
             fig_path=str(os.path.join(wandb.run.dir, f'{run_id}.png')))
    wandb.save(datafile)
    with open(os.path.join(wandb.run.dir, f'pars_{run_id}.pkl'), 'wb') as f:
        pickle.dump(sim.pars, f)
    
if __name__ == '__main__':
    fire.Fire(train)
