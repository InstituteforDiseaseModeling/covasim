'''
Speed and implementation tests
'''

import numpy as np
import sciris as sc
import numba as nb

class NumbaTests(sc.prettyobj):
    def __init__(self, n=100e3, npts=100, keys=None):
        if keys is None:
            keys = ['a','b']
        self.n = int(n)
        self.npts = npts
        self.keys = keys
        self.states = {key:np.random.random(self.n) for key in self.keys}
        self.attr_a = np.random.random(self.n)
        self.attr_b = np.random.random(self.n)
        self.results = np.zeros(self.npts)

    @property
    def prop_a(self):
        return self.states['a']

    @property
    def prop_b(self):
        return self.states['b']

    def prop_next(self, t):
        self.results[t] += (self.prop_a * self.prop_b).sum()
        return

    def attr_next(self, t):
        self.results[t] += (self.attr_a * self.attr_b).sum()
        return

    def prop_run(self):
        for t in range(self.npts):
            self.prop_next(t)
        return

    def attr_run(self):
        for t in range(self.npts):
            self.attr_next(t)
        return


if __name__ == '__main__':

    nt = NumbaTests()

    sc.tic()
    nt.prop_run()
    sc.toc(label='prop')

    sc.tic()
    nt.attr_run()
    sc.toc(label='attr')