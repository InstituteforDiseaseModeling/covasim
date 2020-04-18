import covasim as cv
import sciris as sc

cv.check_version('0.27.9')

do_plot = 1

datafile = sc.thisdir(__file__, '../../example_data.csv')

pars = sc.objdict(
    diag_factor = 1.0,
    n_days=30,
    pop_infected=100
    )


sim = cv.Sim(pars, datafile=datafile)

case = 3
if case == 0: # works, no diagnoses
    testprob = cv.test_prob(symp_prob=0, asymp_prob=0, test_sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0)
elif case == 1: # works, most diagnosed, slight mismatch due to new infections that day
    testprob = cv.test_prob(symp_prob=1, asymp_prob=1, test_sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0)
elif case == 2: # works, no diagnoses
    testprob = cv.test_prob(symp_prob=1, asymp_prob=1,test_sensitivity=0.0, loss_prob=0.0, test_delay=0, start_day=0)
elif case == 3: # works, ~50% diagnosed
    testprob = cv.test_prob(symp_prob=1, asymp_prob=1, test_sensitivity=0.5, loss_prob=0.0, test_delay=0, start_day=0)

sim.update_pars(interventions=testprob)

to_plot = sc.dcp(cv.default_sim_plots)
to_plot['Total counts'].append('cum_infectious')
to_plot['Daily counts']+= ['new_tests', 'new_infectious']
sim.run(do_plot=do_plot, to_plot=to_plot)