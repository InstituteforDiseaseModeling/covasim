import covasim as cv
import sciris as sc


pars = sc.objdict(
    pop_size     = 40e3,    # Population size
    location = "Vorarlberg",
    pop_infected = 10,       # Number of initial infections
    n_days       = 40,       # Number of days to simulate
    pop_scale = 10,
    n_beds_icu = 30,
    n_beds_hosp = 80,
    contacts = 0.5
)

mysim = cv.sim.Sim(pars=pars,load_pop=True, popfile='voriPop.pop')
#multiSim = cv.MultiSim(sims=mysim,n_runs=2)
mysim.run()
mysim.plot()

#mysim.reduce()
mysim.plot()