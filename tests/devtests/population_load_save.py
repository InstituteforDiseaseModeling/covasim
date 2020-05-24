import covasim as cv
import numpy as np
import sciris as sc

T = sc.tic()

seeds = [55, 66]
pop_type = 'random'

lkmap = {'random':'a', 'hybrid':'w', 'synthpops':'w'}
lkey = lkmap[pop_type]

np.random.seed(seeds[0])
s1 = cv.Sim(popfile='p1.pop', save_pop=True, rand_seed=seeds[0], pop_type=pop_type, label='p1')
s1.initialize()
np.random.seed(seeds[0])
s1b = cv.Sim(popfile='p1b.pop', save_pop=True, rand_seed=seeds[0], pop_type=pop_type, label='p1b')
s1b.initialize()
np.random.seed(seeds[1])
s2 = cv.Sim(popfile='p2.pop', save_pop=True, rand_seed=seeds[1], pop_type=pop_type, label='p2')
s2.initialize()
s1c = cv.Sim(popfile='p1.pop', load_pop=True, rand_seed=seeds[0], pop_type=pop_type, label='p1c')
s1c.initialize()


def check_eq(a, b):
    assert np.allclose(a, b, rtol=0, atol=0, equal_nan=True)


for key in s1.people.keys():
    print(f'Checking key {key}...')
    check_eq(s1.people[key], s1b.people[key])
    check_eq(s1.people[key], s1c.people[key])


s1.initialize()
s1.run(verbose=0)
s1b.initialize()
s1b.run(verbose=0)
s2.initialize()
s2.run(verbose=0)
s1c.initialize()
s1c.run(verbose=0)

assert s1.summary == s1b.summary
assert s1.summary == s1c.summary
assert s1.summary != s2.summary
assert (s1.people.contacts[lkey]['p2'] == s1c.people.contacts[lkey]['p2']).all()
assert len(s1.people.contacts) == len(s1b.people.contacts)
assert len(s1.people.contacts) == len(s1c.people.contacts)
assert len(s1.people.contacts) != len(s2.people.contacts)

sc.toc(T)