'''
Speed and implementation tests
'''

import numpy as np
import pylab as pl
import sciris as sc
import numba as nb

spec = [
    ('n', nb.int32),
    ('npts', nb.int32),

]

@nb.jitclass(spec)
class NumbaTests(sc.prettyobj):
    def __init__(self, n=400e3, npts=1000):
        self.n = int(n)
        self.npts = npts
        self.keys = ['a','b']
        self.states = {key:np.random.random(self.n) for key in self.keys}
        self.results = np.zeros(self.npts)

    @property
    def a(self):
        return self.states['a']

    @property
    def b(self):
        return self.states['b']

    def next_mult(self, t):
        self.results[t] += (self.a * self.b).sum()
        return

    def next_cond(self, t):
        self.results[t] += np.logical_and(self.a>0.5, self.b>0.5).sum()
        return

    def run(self):
        for t in range(self.npts):
            self.next_cond(t)
        return


if __name__ == '__main__':

    ns = (1+np.arange(15))*100e3

    ts = []
    for n in ns:
        nt = NumbaTests(n=n)
        sc.tic()
        nt.run()
        t = sc.toc(output=True)
        ts.append(t)
        print(n, t)

    pl.subplot(2,1,1)
    pl.scatter(ns/1e6, ts)
    sc.setylim()
    pl.xlabel('Number of points (millions)')
    pl.ylabel('Calculation time')

    pl.subplot(2,1,2)
    pl.scatter(ns/1e6, np.array(ts)/ns*1e6)
    sc.setylim()
    pl.xlabel('Number of points (millions)')
    pl.ylabel('Calculation time per point (μs)')