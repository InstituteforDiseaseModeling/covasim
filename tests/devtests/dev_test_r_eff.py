"""
Plot time average r_eff
Check that r_eff agrees for first evaluation regardless of window size
"""
import covasim as cv
import pylab as pl
import numpy as np
import sciris as sc


#%% Legacy tests

sim = cv.Sim()
sim.run(verbose=False);

fig, axes = pl.subplots(figsize=(5,4))
window= [7,1]
reff_t0 = []
for iw, w in enumerate(window):
    r_eff = sim.compute_r_eff(method='infectious', window=w)
    axes.plot(sim.tvec, r_eff.values, '-o', label=w)
    reff_t0.append(sim.results['r_eff'].values[np.isfinite(sim.results['r_eff'].values)][0])
axes.legend()
axes.set_xlabel('time (days)')
axes.set_ylabel('r_eff');
pl.tight_layout()

assert len(np.unique(reff_t0)) == 1


#%% New tests

sim = cv.Sim(pop_size=100e3, pop_infected=100, n_days=90)
iday = 50
cb = cv.change_beta(days=iday, changes=0)
sim.update_pars(interventions=cb)
sim.run(verbose=False)

plot_args = dict(lw=3, alpha=0.7)
fig = sim.plot_result('r_eff')
r_eff_d = sc.dcp(sim.results['r_eff'].values)
r_eff_i = sc.dcp(sim.compute_r_eff(method='infectious').values)
r_eff_o = sc.dcp(sim.compute_r_eff(method='outcome').values)
pl.plot(r_eff_i, label='Method from infectious', c=[1.0,0.3,0], **plot_args)
pl.plot(r_eff_o, label='Method from outcome', c=[0,0.5,0.0], **plot_args)
pl.legend()

print(np.nansum(r_eff_d))
print(np.nansum(r_eff_i))
print(np.nansum(r_eff_o))

