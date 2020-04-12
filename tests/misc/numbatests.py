'''
Speed and implementation tests
'''

import numpy as np
import pylab as pl
import sciris as sc
import numba as nb

class NumbaTests(sc.prettyobj):
    def __init__(self, n=400e3, npts=1000, keys=None):
        if keys is None:
            keys = ['a','b']
        self.n = int(n)
        self.npts = npts
        self.keys = keys
        self.states = {key:np.random.random(self.n) for key in self.keys}
        self.results = np.zeros(self.npts)

    @property
    def a(self):
        return self.states['a']

    @property
    def b(self):
        return self.states['b']

    def next(self, t):
        self.results[t] += (self.a * self.b).sum()
        return

    def run(self):
        for t in range(self.npts):
            self.next(t)
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