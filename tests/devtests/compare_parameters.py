import numpy   as np; np
import pylab   as pl; pl
import pandas  as pd; pd
import sciris  as sc; sc
import covasim as cv; cv

pl.rc('figure', dpi=150)


new = sc.objdict()
orig = sc.objdict()

ages = np.arange(0,100,10) + 5

orig.severe_probs  = np.array([0.00050, 0.00165, 0.00720, 0.02080, 0.03430, 0.07650, 0.13280, 0.20655, 0.24570, 0.24570])
orig.crit_probs    = np.array([0.00003, 0.00008, 0.00036, 0.00104, 0.00216, 0.00933, 0.03639, 0.08923, 0.17420, 0.17420])
orig.death_probs   = np.array([0.00002, 0.00006, 0.00030, 0.00080, 0.00150, 0.00600, 0.02200, 0.05100, 0.09300, 0.09300])

new.severe_probs  = np.array([0.00050, 0.00165, 0.0169, 0.0169, 0.036, 0.036, 0.1296, 0.1296, 0.1296, 0.1296])
new.crit_probs    = np.array([0.00003, 0.00008, 0.00036, 0.00104, 0.00216, 0.00933, 0.03639, 0.08923, 0.17420, 0.17420])
new.death_probs   = np.array([2e-05, 2e-05, 9.5e-05, 0.00032, 0.00098, 0.00265, 0.007655, 0.024385, 0.08292, 0.08292])


#%% Plot against each other
fig = pl.figure(figsize=(24,12))


for j,plot_func in enumerate([pl.semilogy, pl.plot]):
    ax1 = pl.subplot(2,3,1+j)
    ax1.set_title('Original')
    for k,v in orig.items():
        plot_func(ages, v, '-o', label=k)

    ax2 = pl.subplot(2,3,4+j)
    ax2.set_title('New')
    for k,v in new.items():
        plot_func(ages, v, '-o', label=k)


    for ax in [ax1, ax2]:
        ax.legend()


for j,data in enumerate([orig, new]):
    label = 'Original' if j==0 else 'New'
    ax = pl.subplot(2,3,3+3*j)
    ax.set_title(label)
    pl.plot(ages, data.crit_probs/data.severe_probs, '-o', label='crit/severe')
    pl.plot(ages, data.death_probs/data.crit_probs, '-o', label='death/crit')
    ax.legend()

pl.show()
