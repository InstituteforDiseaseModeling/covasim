import covasim as cv

n_days = 90
n_tests = 100
delay = 10
tn = cv.test_num(daily_tests=[n_tests]*n_days, test_delay=delay)

sim = cv.Sim(interventions=tn, n_days=n_days, diag_factor=1.0)
sim.run()
sim.plot()