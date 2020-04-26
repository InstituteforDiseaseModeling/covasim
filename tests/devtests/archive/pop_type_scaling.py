import covasim as cv
import numpy as np


s1s = []
s2s = []
s3s = []
ps = [5000, 10000, 20000, 50000, 100000][0]
pi = 100
nd = 30
for i in [1, 2, 3]:
    print(i)
    s1 = cv.Sim(pop_size=ps, n_days=nd, pop_infected=pi, pop_type='random', rand_seed=i, verbose=0)
    s1.run()
    s1s.append(s1.summary['cum_infections'])
    s2 = cv.Sim(pop_size=ps, n_days=nd, pop_infected=pi, pop_type='hybrid', rand_seed=i, verbose=0)
    s2.run()
    s2s.append(s2.summary['cum_infections'])
    s3 = cv.Sim(pop_size=ps, n_days=nd, pop_infected=pi, pop_type='synthpops', rand_seed=i, verbose=0)
    s3.run()
    s3s.append(s3.summary['cum_infections'])

print(s1s)
print(s2s)
print(s3s)
print(np.mean(s1s))
print(np.mean(s2s))
print(np.mean(s3s))

