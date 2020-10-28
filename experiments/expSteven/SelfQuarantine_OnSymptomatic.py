import covasim as cv
from experiments.SimVorarlberg.pars import pars
from experiments.SimVorarlberg.specialInterventions.selfequarantinOnSymptomatic_intervention import selfequarantinOnSymptomatic_intervention

if __name__ == '__main__':

    #pars.interventions = selfequarantinOnSymptomatic_intervention(1,['w','s', 'c'])
#
#
    #sim_pop1 = cv.Sim(pars)
    #sim_pop1.init_people(load_pop=True, popfile='pop1.pop')
#
    #sim_pop2 = cv.Sim(pars)
    #sim_pop2.init_people(load_pop=True, popfile='pop2.pop')
#
    #sim_pop3 = cv.Sim(pars)
    #sim_pop3.init_people(load_pop=True, popfile='pop3.pop')
#
    #multisim1 = cv.MultiSim(sims=sim_pop1, n_runs=11, noise=0.1, keep_people=True, quantiles={'low':0.4, 'high':0.6})
    #multisim2 = cv.MultiSim(sims=sim_pop2, n_runs=11, noise=0.1, keep_people=True, quantiles={'low':0.4, 'high':0.6})
    #multisim3 = cv.MultiSim(sims=sim_pop3, n_runs=11, noise=0.1, keep_people=True, quantiles={'low':0.4, 'high':0.6})
#
    #multisim1.run()
    #multisim2.run()
    #multisim3.run()
#
    #mergedSim = cv.MultiSim.merge([multisim1,multisim2,multisim3])
    #mergedSim.reduce()
    #mergedSim.plot()
#
    #mergedSim.save("SO_OS2.msim")
    sim = cv.MultiSim.load('SO_OS2.msim')
    sim.plot()