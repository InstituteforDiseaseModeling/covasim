'''
Confirming that reset works
'''

import covasim as cv
import pytest

sim = cv.Sim()
r1 = sim.run()

# Rerunning without resetting is an error
with pytest.raises(RuntimeError):
    sim.run()

# Should get identical results after reset
sim.reset()
r2 = sim.run()
assert (r1['cum_infections'].values == r2['cum_infections'].values).all()