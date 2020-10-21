import covasim as cv
from SimVorarlberg.pars import pars

if __name__ == '__main__':

    sim = cv.Sim(pars)
    sim.init_people(load_pop=True,popfile='testPop.pop')
    sim.run(do_plot=True, restore_pars=True, reset_seed=True)