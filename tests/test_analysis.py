'''
Execute analysis tools in order to broadly cover basic functionality of analysis.py
'''

import covasim as cv
import numpy as np


# Runtime: 5.436 seconds

default_pop_size = 1000
# Tests that snapshot options are accessing correct data
def test_snapshot():
    sim = cv.Sim(pop_size = default_pop_size, analyzers=cv.snapshot('2020-04-04', '2020-04-14'))
    sim.run()
    snapshot = sim['analyzers'][0]
    people1 = snapshot.snapshots[0]            # Option 1
    people2 = snapshot.snapshots['2020-04-04'] # Option 2
    people3 = snapshot.get('2020-04-04')       # Option 3
    people4 = snapshot.get(34)                 # Option 4
    people5 = snapshot.get()                   # Option 5

    peoples = [people2, people3, people4, people5]
    for i, people in enumerate(peoples):
        optionNum = i+2
        assert people1 == people, f"Option {optionNum} is not accessing correct date"


# Tests that histogram analyzer attaches and reports correctly
def test_hist():
    # raising multiple histograms to check windows functionality
    day_list = ["2020-03-30", "2020-03-31", "2020-04-01"]
    age_analyzer = cv.age_histogram(days=day_list)
    sim = cv.Sim(pop_size = default_pop_size, analyzers=age_analyzer)
    sim.run()


    # checks to see that compute windows returns correct number of results
    agehist = sim['analyzers'][0]
    agehist.compute_windows()
    assert len(age_analyzer.window_hists) == len(day_list), "Number of histograms should equal number of days"

    # checks compute_windows and plot()
    plots = agehist.plot(windows=True)  # .savefig('DEBUG_age_histograms.png')
    assert len(plots) == len(day_list), "Number of plots generated should equal number of days"

# Tests that fit object computes statistics without fail
def test_fit():
    sim = cv.Sim(rand_seed=1, pop_size = default_pop_size, datafile="example_data.csv")
    sim.run()

    # checking that Fit can handle custom input
    custom_inputs = {'BoomTown':{'data':np.array([1,2,3]), 'sim':np.array([1,2,4]), 'weights':[2.0, 3.0, 4.0]}}
    fit = sim.compute_fit(custom=custom_inputs, compute=True)
    
    # tests that different seed will change compute results
    sim2 = cv.Sim(rand_seed=2, pop_size = default_pop_size, datafile="example_data.csv")
    sim2.run()
    otherFit = sim2.compute_fit(custom=custom_inputs)

    assert fit.mismatch != otherFit.mismatch, f"Differences between fit and data remains unchanged after changing sim seed"

#%% Run as a script
if __name__ == '__main__':
    import time
    start_time = time.time()

    test_snapshot()
    test_hist()
    test_fit()

    print("--- %s seconds ---" % (time.time() - start_time))




