'''
Example for running Optuna
'''

import os
import sciris as sc
import covasim as cv
import optuna as op

# Create a (mutable) dictionary for global settings
g = sc.objdict()
g.name      = 'my-example-calibration'
g.db_name   = f'{g.name}.db'
g.storage   = f'sqlite:///{g.db_name}'
g.n_workers = 4 # Define how many workers to run in parallel
g.n_trials = 25 # Define the number of trials, i.e. sim runs, per worker


def run_sim(pars, label=None, return_sim=False):
    ''' Create and run a simulation '''
    print(f'Running sim for beta={pars["beta"]}, rel_death_prob={pars["rel_death_prob"]}')
    pars = dict(
        pop_size       = 10_000,
        start_day      = '2020-02-01',
        end_day        = '2020-04-11',
        beta           = pars["beta"],
        rel_death_prob = pars["rel_death_prob"],
        interventions  = cv.test_num(daily_tests='data'),
        verbose        = 0,
    )
    sim = cv.Sim(pars=pars, datafile='example_data.csv', label=label)
    sim.run()
    fit = sim.compute_fit()
    if return_sim:
        return sim
    else:
        return fit.mismatch


def run_trial(trial):
    ''' Define the objective for Optuna '''
    pars = {}
    pars["beta"]           = trial.suggest_uniform('beta', 0.005, 0.020) # Sample from beta values within this range
    pars["rel_death_prob"] = trial.suggest_uniform('rel_death_prob', 0.5, 3.0) # Sample from beta values within this range
    mismatch = run_sim(pars)
    return mismatch


def worker():
    ''' Run a single worker '''
    study = op.load_study(storage=g.storage, study_name=g.name)
    output = study.optimize(run_trial, n_trials=g.n_trials)
    return output


def run_workers():
    ''' Run multiple workers in parallel '''
    output = sc.parallelize(worker, g.n_workers)
    return output


def make_study():
    ''' Make a study, deleting one if it already exists '''
    if os.path.exists(g.db_name):
        os.remove(g.db_name)
        print(f'Removed existing calibration {g.db_name}')
    output = op.create_study(storage=g.storage, study_name=g.name)
    return output


if __name__ == '__main__':

    # Run the optimization
    t0 = sc.tic()
    make_study()
    run_workers()
    study = op.load_study(storage=g.storage, study_name=g.name)
    best_pars = study.best_params
    T = sc.toc(t0, output=True)
    print(f'Output: {best_pars}, time: {T}')

    # Plot the results
    initial_pars = dict(beta=0.015, rel_death_prob=1.0)
    before = run_sim(pars=initial_pars, label='Before calibration', return_sim=True)
    after = run_sim(pars=best_pars, label='After calibration', return_sim=True)
    msim = cv.MultiSim([before, after])
    msim.plot(to_plot=['cum_tests', 'cum_diagnoses', 'cum_deaths'])