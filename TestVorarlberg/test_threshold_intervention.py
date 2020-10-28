import covasim as cv
from experiments.SimVorarlberg.pars import pars
from experiments.SimVorarlberg.specialInterventions.threshold_intervention import threshold_intervention

if __name__ == '__main__':

    intervention_over_th = cv.change_beta(0, 0.3)
    intervention_under_th = cv.change_beta(0, 1)

    th_intervention = threshold_intervention('n_severe', 20, intervention_over_th, intervention_under_th)
    pars.interventions = th_intervention

    sim = cv.Sim(pars=pars)
    sim.init_people(load_pop=True,popfile='testPop.pop')
    sim.initialize()
    sim.run(verbose=1, restore_pars=True, reset_seed=True)
    sim.plot()