''' Compute doubling time and generate plots '''

import sciris as sc
import numpy as np
import pylab as pl
import statsmodels.api as sm
import covid_seattle

fn = 'seattle-projection-baseline_v4e.obj'
burnin = 10
scale = pars['scale']

def compute_doubling_time(sims, burnin = 10, verbose = False):
    '''Compute doubling time from simulation data.  Note - assumes a one-day timestep

    :param sims: Matrix with one row per day, one column per realization
    :param burning: The number of days to reject from beginning of the series
    :param verbose: Set True to see regression summary
    '''

    T = np.arange(len(sims[var]))

    T_poststart = np.arange(len(sims[var][burnin:]))
    T_poststart_all = np.repeat(T_poststart, sims[var][burnin:].shape[1])

    Y_poststart = sims[var][burnin:].flatten()

    exog = sm.add_constant(T_poststart_all)
    endog = np.log2(Y_poststart)

    # Ordinary least squares on log2 transformed data
    model = sm.OLS(endog, exog)

    results = model.fit()
    if verbose: print(results.summary())

    doubling_time = 1/results.params[1]

    return doubling_time


pars = covid_seattle.make_pars() # TODO: should be gotten from a sim

sims = sc.loadobj(fn) # Load in the simulations

var = 'cum_exposed' #'n_exposed' # <-- Select which variable to analyze
sims[var] *= scale # Increase by scale factor

doubling_time = compute_doubling_time(sims):


fig = pl.figure(figsize=(16,8))
ax = pl.subplot(2,1,1)
pl.grid(True)
ax.plot(sims[var])
ax.plot( burnin + T_poststart, 2**(results.params[0] + results.params[1]*T_poststart), 'k--', lw=3)
ax.set_ylabel('Cumulative Infections')


ax = pl.subplot(2,1,2)
pl.grid(True)
ax.plot(sims[var])
ax.plot( burnin + T_poststart, 2**(results.params[0] + results.params[1]*T_poststart), 'k--', lw=3)

ax.set_yscale('log', basey=2)

ax.set_xlabel('Day')
ax.set_ylabel('Cumulative Infections (log2-scale)')

print(f'Doubling time is {doubling_time}')
pl.suptitle(f'Doubling time: {doubling_time:.2f}', fontsize=20)
pl.savefig('seattle-projection-baseline_v4e.png')

pl.show()

print('Done.')
