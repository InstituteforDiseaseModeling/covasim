import covasim as cv
import pytest

with pytest.raises(ValueError):
    s1 = cv.Sim(start_day='2020-06-01', end_day='2020-04-01')
    s1.run()

with pytest.raises(ValueError):
    s2 = cv.Sim(end_day=None, n_days=None)
    s2.run()

sim = cv.Sim(pop_size=100e3, start_day='2020-01-01', end_day='2020-04-01')
sim.run()