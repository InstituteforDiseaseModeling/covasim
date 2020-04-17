"""
Plot time average r_eff
Check that r_eff agrees for first evaluation regardless of window size
"""
import covasim as cv
import matplotlib.pyplot as plt
import numpy as np

sim = cv.Sim()
sim.run(verbose=False);

fig, axes = plt.subplots(figsize=(5,4))
window= [7,1]
reff_t0 = []
for iw, w in enumerate(window):
    sim.compute_r_eff(window=w)
    axes.plot(sim.tvec, sim.results['r_eff'].values, '-o', label=w)
    reff_t0.append(sim.results['r_eff'].values[np.isfinite(sim.results['r_eff'].values)][0])
axes.legend()
axes.set_xlabel('time (days)')
axes.set_ylabel('r_eff');
plt.tight_layout()
plt.show()

assert len(np.unique(reff_t0)) == 1