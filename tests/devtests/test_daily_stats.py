'''
Test daily stats analyzer
'''

import covasim as cv

tp = cv.test_prob(symp_prob=0.1)
cb = cv.change_beta(days=0.5, changes=0.3, label='NPI')
sim = cv.Sim(interventions=[tp, cb], analyzers=cv.daily_stats())
sim.run()