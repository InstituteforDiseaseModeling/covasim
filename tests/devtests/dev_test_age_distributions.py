'''
Plot age distributions for Sweden and Somalia as a test.
'''

import covasim as cv
import pylab as pl

loc1 = 'Sweden'
loc2 = 'Somalia'

sim1 = cv.Sim(location=loc1)
sim2 = cv.Sim(location=loc2)

sim1.initialize()
sim2.initialize()

ages1 = sim1.people.age
ages2 = sim2.people.age

n = 100
fig = pl.figure()

pl.subplot(2,1,1)
pl.hist(ages1, n)
pl.title(loc1)

pl.subplot(2,1,2)
pl.hist(ages2, n)
pl.title(loc2)
pl.xlabel('Age')
pl.ylabel('Count')