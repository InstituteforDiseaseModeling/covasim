import covasim as cv
from experiments.SimVorarlberg.pars import pars
import sciris as sc

if __name__ == "__main__":

    filepath = sc.makefilepath(filename='pop1.pop')

    pop1_sim = cv.Sim(pars=pars, popfile=filepath, load_pop=True)
    pop1_multisim = cv.MultiSim(sims=pop1_sim, n_runs=11, noise=0.0, keep_people=True, quantiles={'low':0.1, 'high':0.9})

    filepath = sc.makefilepath(filename='pop2.pop')

    pop2_sim = cv.Sim(pars=pars, popfile=filepath, load_pop=True)
    pop2_multisim = cv.MultiSim(sims=pop2_sim, n_runs=11, noise=0.0, keep_people=True, quantiles={'low':0.1, 'high':0.9})

    filepath = sc.makefilepath(filename='pop3.pop')

    pop3_sim = cv.Sim(pars=pars, popfile=filepath, load_pop=True)
    pop3_multisim = cv.MultiSim(sims=pop3_sim, n_runs=11, noise=0.0, keep_people=True, quantiles={'low':0.1, 'high':0.9})

    pop1_multisim.run()
    pop2_multisim.run()
    pop3_multisim.run()

    pop1_multisim.save("pop1.msim")
    pop2_multisim.save("pop2.msim")
    pop3_multisim.save("pop3.msim")

    pop1_multisim.reduce()
    pop2_multisim.reduce()
    pop3_multisim.reduce()

    pop1_multisim.plot()
    pop2_multisim.plot()
    pop3_multisim.plot()