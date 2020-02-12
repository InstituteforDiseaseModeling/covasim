# Simple example usage for the Covid-19 agent-based model

import covid_abm

doplot = True
dosave = False

sim = covid_abm.Sim()
sim.run()
if doplot:
    sim.plot(dosave=dosave)

print('Done.')

