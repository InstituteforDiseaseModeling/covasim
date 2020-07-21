import covasim as cv
import numpy as np


# Suite of tests to test basic functionality of the analysis.py file
# Runtime: 5.004 seconds
def test_analysis_snapshot():
    sim = cv.Sim(analyzers=cv.snapshot('2020-04-04', '2020-04-14'))
    sim.run()
    snapshot = sim['analyzers'][0]
    people1 = snapshot.snapshots[0]            # Option 1
    people2 = snapshot.snapshots['2020-04-04'] # Option 2
    people3 = snapshot.get('2020-04-14')       # Option 3
    people4 = snapshot.get(34)                 # Option 4
    people5 = snapshot.get()                   # Option 5
    # people3 = []  # uncomment to verify error
    peoples = [people1, people2, people3, people4, people5]
    for i, people in enumerate(peoples):
        optionNum = i+1
        assert len(people) > 0, f"Option {optionNum} should have more than 0 members"



def test_analysis_hist():
    # raising multiple histograms to check windows functionality
    day_list = ["2020-03-30", "2020-03-31", "2020-04-01"]
    age_analyzer = cv.age_histogram(days=day_list)
    sim = cv.Sim(analyzers=age_analyzer)
    sim.run()
    assert age_analyzer.window_hists == None

    # checks to make sure dictionary form has right keys
    agehistDict = sim['analyzers'][0].get()
    assert len(agehistDict.keys()) == 5 # TODO: is there a way to know what keys to expect?

    # checks to see that compute windows is correct
    agehist = sim['analyzers'][0]
    agehist.compute_windows()
    assert len(age_analyzer.window_hists) == len(day_list), "Number of histograms should equal number of days"

    # checks compute_windows and plot()
    plots = agehist.plot(windows=True)  # .savefig('DEBUG_age_histograms.png')
    assert len(plots) == len(day_list), "Number of plots generated should equal number of days"


def test_analysis_fit():
    sim = cv.Sim(datafile="example_data.csv")
    sim.run()
    fit = sim.compute_fit()
    print(type(fit))
    # battery of tests to test basic fit function functionality
    # tests that running functions does not produce error
    # try/except blocks will be changed to assertRaises once code is formatted as unittest
    try:
        fit.compute_losses()
    except Exception as E:
        raise ValueError(f"Unable to compute losses: {E}")
    try:
        fit.compute_diffs()
    except Exception as E:
        raise ValueError(f"Unable to compute differences: {E}")
    try:
        fit.compute_gofs()
    except Exception as E:
        raise ValueError(f"Unable to compute goodness of fit: {E}")
    try:
        fit.plot()
    except Exception as E:
        raise ValueError(f"Fit plot not being rendered correctly: {E}")

    # testing custom fit outputs with new data
    # expected: added data will change outputs
    initial_gofs = fit.gofs
    initial_losses = fit.losses
    initial_diffs = fit.diffs
    customInputs = {'BoomTown':{'data':np.array([1,2,3]), 'sim':np.array([1,2,4]), 'weights':[2.0, 3.0, 4.0]}}
    try:
        customFit = sim.compute_fit(custom=customInputs)
    except:
        raise ValueError("Fitting the model does not work with custom inputs")

    new_gofs = customFit.gofs
    new_losses = customFit.losses
    new_diffs = customFit.diffs
    assert initial_gofs != new_gofs, f"Goodness of fit remains unchanged after adding new data"
    assert initial_losses != new_losses, f"Losses between data and fit remain unchanged after adding new data"
    assert initial_diffs != new_diffs, f"Differences between data and fit remains unchanged after adding new data"

    # testing that customFit is different from 
    # TODO: test the following `customFit.reconcile_inputs()`
    # TODO: test difference between data and sim, ensure loss is

def test_trans_tree():
    sim = cv.Sim()
    sim.run()
    # testing that it catches no graph error
    tt = sim.make_transtree(to_networkx=False)
    try:
        tt.r0()
    except RuntimeError:
        pass

# uncomment to check runtime :
# import time
# start_time = time.time()

# test_analysis_snapshot()
# test_analysis_hist()
# test_analysis_fit()
# test_trans_tree()

# print("--- %s seconds ---" % (time.time() - start_time))

