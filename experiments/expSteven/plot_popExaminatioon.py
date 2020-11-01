import covasim as cv

if __name__ == '__main__':

    msim1 = cv.MultiSim.load('pop1.msim')
    msim2 = cv.MultiSim.load('pop2.msim')
    msim3 = cv.MultiSim.load('pop3.msim')

    msim1.reduce()
    msim2.reduce()
    msim3.reduce()

    msim1.plot({'Cumulative infections': ['cum_infections'], 'Health outcomes' : ['cum_severe', 'cum_critical', 'cum_deaths']})
    msim2.plot({'Cumulative infections': ['cum_infections'], 'Health outcomes' : ['cum_severe', 'cum_critical', 'cum_deaths']})
    msim3.plot({'Cumulative infections': ['cum_infections'], 'Health outcomes' : ['cum_severe', 'cum_critical', 'cum_deaths']})
