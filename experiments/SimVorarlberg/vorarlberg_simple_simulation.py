import covasim as cv
from SimVorarlberg.pars import pars

if __name__ == '__main__':

    sim = cv.Sim(pars)
    sim.popdict = cv.load('testPop.pop')
    sim.run()