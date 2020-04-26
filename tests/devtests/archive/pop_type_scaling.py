import covasim as cv
import numpy as np


s1s = []
s2s = []
s3s = []
bl = dict(h=1.3,s=0.7,w=0.7,c=0.2)
# f = 1.7
# sbl = dict(h=f*1.4,s=f*0.7,w=f*0.7,c=f*0.2)
sbl = dict(h=2.0, s=1.0, w=1.0, c=0.3)
ps = 100000
pi = 20
nd = 30
for i in [1, 2]:
    print(i)
    s1 = cv.Sim(pop_size=ps, n_days=nd, pop_infected=pi, pop_type='random', rand_seed=i, verbose=0)
    s1.run()
    s1s.append(s1.summary['cum_infections'])
    s2 = cv.Sim(pop_size=ps, n_days=nd, pop_infected=pi, pop_type='hybrid', rand_seed=i, verbose=0, beta_layer=sbl)
    s2.run()
    s2s.append(s2.summary['cum_infections'])
    s3 = cv.Sim(pop_size=ps, n_days=nd, pop_infected=pi, pop_type='synthpops', rand_seed=i, verbose=0, beta_layer=sbl)
    s3.run()
    s3s.append(s3.summary['cum_infections'])

print(s1s)
print(s2s)
print(s3s)
print(np.mean(s1s))
print(np.mean(s2s))
print(np.mean(s3s))
# sim.reset_layer_pars()
# sim.update_pars()

