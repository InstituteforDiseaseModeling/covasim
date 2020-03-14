import coverage

cov = coverage.Coverage(source=["covid_seattle","covid_abm"])
cov.start()

import test_age_structure
import test_parameters
import test_sim
import test_utils


test_age_structure.test_age_structure()

test_parameters.test_parameters()
test_parameters.test_data()
test_parameters.test_age_sex(do_plot=False)

# test_sim.test_multiscale()
test_sim.test_parsobj()
test_sim.test_sim(doplot=False)
test_sim.test_trans_tree(doplot=False)
# call some code

cov.stop()
cov.save()

cov.html_report()

