import covasim as cv
from experiments.SimVorarlberg.pars import pars
from experiments.SimVorarlberg.specialInterventions.selfequarantinOnSymptomatic_intervention import selfequarantinOnSymptomatic_intervention

if __name__ == '__main__':

    pars.interventions = cv.test_prob(start_day=0, test_delay=1, test_sensitivity=0.98, symp_prob=1)

    sim = cv.Sim(pars)
    sim.init_people(load_pop=True,popfile='testPop.pop')
    sim.run(do_plot=True, restore_pars=True, reset_seed=True)