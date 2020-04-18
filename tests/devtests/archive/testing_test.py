import covasim as cv
import sciris as sc

do_plot = 1

datafile = sc.thisdir(__file__, '../../example_data.csv')



pars = sc.objdict(
    diag_factor = 1.0,
    n_days=30,
    pop_infected=100
    )


sim = cv.Sim(pars, datafile=datafile)


choice = 2
if choice == 1:
    testnum = cv.test_num(daily_tests=[1000]*60)
elif choice == 2:
    testnum = cv.test_num(daily_tests=sim.data['new_tests'])

sim.update_pars(interventions=[testnum])

sim.run(do_plot=do_plot)