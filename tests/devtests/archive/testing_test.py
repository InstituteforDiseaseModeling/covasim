import covasim as cv
import sciris as sc

do_plot = 1

datafile = sc.thisdir(__file__, '../../example_data.csv')

testnum = cv.test_num(daily_tests=[1000]*60)

pars = sc.objdict(
    # beta = 0.0
    diag_factor = 1.0,
    interventions=testnum,
    n_days=30,
    pop_infected=100
    )


sim = cv.Sim(pars, datafile=datafile)

sim.run(do_plot=do_plot)