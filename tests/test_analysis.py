'''
Execute analysis tools in order to broadly cover basic functionality of analysis.py
'''

import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv


#%% General settings

do_plot = 1

pars = dict(
    pop_size = 1000,
    verbose = 0,
    )


#%% Define tests

def test_snapshot():
    sc.heading('Testing snapshot analyzer')
    sim = cv.Sim(pars, analyzers=cv.snapshot('2020-04-04', '2020-04-14'))
    sim.run()
    snapshot = sim['analyzers'][0]
    people1 = snapshot.snapshots[0]            # Option 1
    people2 = snapshot.snapshots['2020-04-04'] # Option 2
    people3 = snapshot.get('2020-04-14')       # Option 3
    people4 = snapshot.get(34)                 # Option 4
    people5 = snapshot.get()                   # Option 5

    assert people1 == people2, f'Snapshot options should match but do not'
    assert people3 != people4, f'Snapshot options should not match but do'
    return people5


def test_age_hist():
    sc.heading('Testing age histogram')

    day_list = ["2020-03-20", "2020-04-20"]
    age_analyzer = cv.age_histogram(days=day_list)
    sim = cv.Sim(pars, analyzers=age_analyzer)
    sim.run()

    # Checks to see that compute windows returns correct number of results
    agehist = sim['analyzers'][0]
    agehist.compute_windows()
    assert len(age_analyzer.window_hists) == len(day_list), "Number of histograms should equal number of days"

    # checks compute_windows and plot()
    plots = agehist.plot(windows=True)
    assert len(plots) == len(day_list), "Number of plots generated should equal number of days"

    return agehist


def test_fit():
    sc.heading('Testing fitting function')

    # Create a testing intervention to ensure some fit to data
    tp = cv.test_prob(0.1)

    sim = cv.Sim(pars, rand_seed=1, interventions=tp, datafile="example_data.csv")
    sim.run()

    # Checking that Fit can handle custom input
    custom_inputs = {'custom_data':{'data':np.array([1,2,3]), 'sim':np.array([1,2,4]), 'weights':[2.0, 3.0, 4.0]}}
    fit1 = sim.compute_fit(custom=custom_inputs, compute=True)

    # Test that different seed will change compute results
    sim2 = cv.Sim(pars, rand_seed=2, interventions=tp, datafile="example_data.csv")
    sim2.run()
    fit2 = sim2.compute_fit(custom=custom_inputs)

    assert fit1.mismatch != fit2.mismatch, f"Differences between fit and data remains unchanged after changing sim seed"

    return fit1


def test_transtree():
    sc.heading('Testing transmission tree')

    sim = cv.Sim(pars, pop_size=100)
    sim.run()

    transtree = sim.make_transtree()
    transtree.plot()
    transtree.animate(animate=False)
    transtree.plot_histograms()

    return transtree


#%% Run as a script
if __name__ == '__main__':

    # We need to create plots to test plotting, but can use a non-GUI backend
    if not do_plot:
        pl.switch_backend('agg')

    T = sc.tic()

    snapshot  = test_snapshot()
    agehist   = test_age_hist()
    fit       = test_fit()
    transtree = test_transtree()

    print('\n'*2)
    sc.toc(T)
    print('Done.')
