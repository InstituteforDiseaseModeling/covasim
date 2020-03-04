'''
Plot likelihood for COVID-19 cruise ship simulation
'''

#%% Housekeeping

import pylab as pl
import sciris as sc
import covid_abm

r_vec = pl.linspace(0.01, 0.10, 10)
i_vec = pl.arange(2, 11)
n_r = len(r_vec)
n_incub = len(i_vec)


#%% Run simulations

sc.tic()

def run_sim(args):
    sim = covid_abm.Sim()
    sim.pars['r_contact'] = args.r
    sim.pars['incub'] = args.incub
    loglike = sim.likelihood(verbose=0)
    output = sc.objdict({'i':args.i, 'j':args.j, 'loglike':loglike})
    return output

arglist = []
results = pl.zeros((n_r, n_incub))
for i,r in enumerate(r_vec):
    for j,incub in enumerate(i_vec):
        args = sc.objdict({'i':i, 'j':j, 'r':r, 'incub':incub})
        arglist.append(args)

tmp_results = sc.parallelize(run_sim, iterarg=arglist)
for tmp in tmp_results:
    results[tmp.i,tmp.j] = tmp.loglike


sc.toc()

#%% Plotting
pl.figure(figsize=(12,8))
delta_r = (r_vec[1] - r_vec[0])/2
delta_i = (i_vec[1] - i_vec[0])/2
plot_r_vec = pl.hstack([r_vec - delta_r, r_vec[-1]+delta_r])*30*3 # TODO: estimate better from sim
plot_i_vec = pl.hstack([i_vec - delta_i, i_vec[-1]+delta_i])
pl.pcolormesh(plot_i_vec, plot_r_vec, results, cmap=sc.parulacolormap())
# pl.imshow(results)
pl.colorbar()
pl.title('Log-likelihood')
pl.xlabel('Days from exposure to infectiousness')
pl.ylabel('R0')

max_like_ind = pl.argmax(results)
indices = pl.unravel_index(max_like_ind, results.shape)
pl.scatter(indices[0], indices[1], marker='*', s=100, c='black', label='MLE')
pl.legend()
pl.savefig('log-likelihood-example.png')


print('Done.')
