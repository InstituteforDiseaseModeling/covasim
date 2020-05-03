'''
Demonstrate the compare method of multisims
'''

import covasim as cv

s0 = cv.Sim(label='Normal beta')
s1 = cv.Sim(label='Low beta', beta=0.012)
s2 = cv.Sim(label='High beta', beta=0.018)

msim = cv.MultiSim(sims=[s0, s1, s2])
msim.run()
df = msim.compare()

