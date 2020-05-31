'''
Tests for things that are not tested in other files, typically because they are
corner cases or otherwise not part of major workflows.
'''

#%% Imports and settings
import os
import pytest
import sciris as sc
import covasim as cv
import pylab as pl

do_plot = 1
verbose = 0
debug   = 1 # This runs without parallelization; faster with pytest
csv_file  = os.path.join(sc.thisdir(__file__), 'example_data.csv')
xlsx_file = os.path.join(sc.thisdir(__file__), 'example_data.xlsx')


def remove_files(*args):
    ''' Remove files that were created '''
    for path in args:
        if os.path.exists(path):
            print(f'Removing {path}')
            os.remove(path)
    return


#%% Define the tests

def test_base():
    sc.heading('Testing base.py...')

    json_path = 'base_tests.json'
    sim_path  = 'base_tests.sim'

    # Create a small sim for later use
    sim = cv.Sim(pop_size=100, verbose=verbose)
    sim.run()

    # Check setting invalid key
    with pytest.raises(sc.KeyNotFoundError):
        po = cv.ParsObj(pars={'a':2, 'b':3})
        po.update_pars({'c':4})

    # Printing result
    r = cv.Result()
    print(r)
    print(r.npts)

    # Day and date conversion
    daystr = '2020-04-04'
    sim.day(daystr)
    sim.day(sc.readdate(daystr))
    with pytest.raises(ValueError):
        sim.day('not a date')
    sim.date(34)
    sim.date([34, 54])
    sim.date(34, 54, as_date=True)

    # BaseSim methods
    sim.copy()
    sim.export_results(filename=json_path)
    sim.export_pars(filename=json_path)
    sim.shrink(in_place=False)
    for keep_people in [True, False]:
        sim.save(filename=sim_path, keep_people=keep_people)
    cv.Sim.load(sim_path)

    # BasePeople methods
    ppl = sim.people
    ppl.get(['susceptible', 'infectious'])
    ppl.keys()
    ppl.person_keys()
    ppl.state_keys()
    ppl.date_keys()
    ppl.dur_keys()
    ppl.indices()
    ppl._resize_arrays(pop_size=200) # This only resizes the arrays, not actually create new people
    ppl._resize_arrays(pop_size=100) # Change back
    ppl.to_df()
    ppl.to_arr()
    ppl.person(50)
    people = ppl.to_people()
    ppl.from_people(people)
    ppl.make_edgelist([{'new_key':[0,1,2]}])

    # Contacts methods
    contacts = ppl.contacts
    df = contacts['a'].to_df()
    ppl.remove_duplicates(df)
    with pytest.raises(sc.KeyNotFoundError):
        contacts['invalid_key']
    contacts.values()
    len(contacts)

    # Transmission tree methods
    transtree = sim.make_transtree()
    transtree.plot()
    transtree.animate(animate=False)
    transtree.plot_histograms()

    # Tidy up
    remove_files(json_path, sim_path)

    return


def test_interventions():
    sc.heading('Testing interventions')

    # Create sim
    sim = cv.Sim(pop_size=100, n_days=60, datafile=csv_file, verbose=verbose)

    # Intervention conversion
    ce = cv.InterventionDict(**{'which': 'clip_edges', 'pars': {'days': [10, 30], 'changes': [0.5, 1.0]}})
    print(ce)
    with pytest.raises(sc.KeyNotFoundError):
        cv.InterventionDict(**{'which': 'invalid', 'pars': {'days': 10, 'changes': 0.5}})

    # Test numbers and contact tracing
    tn1 = cv.test_num(10, start_day=3, end_day=20)
    tn2 = cv.test_num(daily_tests=sim.data['new_tests'])
    ct = cv.contact_tracing()

    # Create and run
    sim['interventions'] = [ce, tn1, tn2, ct]
    sim.run()

    return



def test_misc():
    sc.heading('Testing miscellaneous functions')

    sim_path = 'test_misc.sim'
    json_path = 'test_misc.json'

    # Data loading
    cv.load_data(csv_file)
    cv.load_data(xlsx_file)

    with pytest.raises(NotImplementedError):
        cv.load_data('example_data.unsupported_extension')

    with pytest.raises(ValueError):
        cv.load_data(xlsx_file, columns=['missing_column'])

    # Dates
    d1 = cv.date('2020-04-04')
    d2 = cv.date(sc.readdate('2020-04-04'))
    ds = cv.date('2020-04-04', d2)
    assert d1 == d2
    assert d2 == ds[0]

    with pytest.raises(ValueError):
        cv.date([(2020,4,4)]) # Raises a TypeError which raises a ValueError

    with pytest.raises(ValueError):
        cv.date('Not a date')

    cv.daydiff('2020-04-04')

    # Saving and loading
    sim = cv.Sim()
    cv.save(filename=sim_path, obj=sim)
    cv.load(filename=sim_path)

    # Version checks
    cv.check_version('0.0.0') # Nonsense version
    print('↑ Should complain about version')
    with pytest.raises(ValueError):
        cv.check_version('0.0.0', die=True)

    # Git checks
    cv.git_info(json_path)
    cv.git_info(json_path, check=True)

    # Poisson tests
    c1 = 5
    c2 = 8
    for alternative in ['two-sided', 'larger', 'smaller']:
        cv.poisson_test(c1, c2, alternative=alternative)
    for method in ['score', 'wald', 'sqrt', 'exact-cond']:
        cv.poisson_test(c1, c2, method=method)

    with pytest.raises(ValueError):
        cv.poisson_test(c1, c2, method='not a method')

    # Tidy up
    remove_files(sim_path, json_path)

    return


def test_people():
    sc.heading('Testing people')

    # Test dynamic layers
    sim = cv.Sim(pop_size=100, n_days=10, verbose=verbose, dynam_layer={'a':1})
    sim.run()

    return


def test_plotting():
    sc.heading('Testing plotting')

    fig_path = 'plotting_test.png'

    # Create sim with data and interventions
    ce = cv.clip_edges(**{'days': 10, 'changes': 0.5})
    sim = cv.Sim(pop_size=100, n_days=60, datafile=csv_file, interventions=ce, verbose=verbose)
    sim.run(do_plot=True)

    # Handle lesser-used plotting options
    sim.plot(to_plot=['cum_deaths', 'new_infections'], sep_figs=True, font_family='Arial', log_scale=['Number of new infections'], interval=5, do_save=True, fig_path=fig_path)
    print('↑ May print a warning about zero values')

    # Handle Plotly functions
    cv.plotly_sim(sim)
    cv.plotly_people(sim)
    cv.plotly_animate(sim)

    # Tidy up
    remove_files(fig_path)

    return


def test_population():
    sc.heading('Testing the population')

    pop_path = 'pop_test.pop'

    # Test locations, including ones that don't work
    cv.Sim(pop_size=100, pop_type='hybrid', location='nigeria').initialize()
    cv.Sim(pop_size=100, pop_type='hybrid', location='not_a_location').initialize()
    print('↑ Should complain about location not found')
    cv.Sim(pop_size=100, pop_type='random', location='lithuania').initialize()
    print('↑ Should complain about missing h layer')

    # Test synthpops
    try:
        sim = cv.Sim(pop_size=500, pop_type='synthpops')
        sim.initialize()
    except Exception as E:
        errormsg = f'Synthpops test did not pass:\n{str(E)}\nNote: synthpops is optional so this exception is OK.'
        print(errormsg)

    # Not working
    with pytest.raises(ValueError):
        sim = cv.Sim(pop_type='not_an_option')
        sim.initialize()

    # Save/load
    sim = cv.Sim(pop_size=100, popfile=pop_path, save_pop=True)
    sim.initialize()
    cv.Sim(pop_size=100, popfile=pop_path, load_pop=True)
    with pytest.raises(ValueError):
        cv.Sim(pop_size=101, popfile=pop_path, load_pop=True)

    remove_files(pop_path)

    return



def test_requirements():
    sc.heading('Testing requirements')

    cv.requirements.min_versions['sciris'] = '99.99.99'
    with pytest.raises(ImportError):
        cv.requirements.check_sciris()

    cv.requirements.min_versions['scirisweb'] = '99.99.99'
    cv.requirements.check_scirisweb()
    with pytest.raises(ImportError):
        cv.requirements.check_scirisweb(die=True)

    cv.requirements.check_synthpops()

    print('↑ Should print various requirements warnings')

    return


def test_run():
    sc.heading('Testing run')

    msim_path  = 'run_test.msim'
    scens_path = 'run_test.scens'

    # Test creation
    s1 = cv.Sim(pop_size=100)
    s2 = s1.copy()
    msim = cv.MultiSim(sims=[s1, s2])
    with pytest.raises(TypeError):
        cv.MultiSim(sims='not a sim')

    # Test other properties
    len(msim)
    msim.result_keys()
    msim.base_sim = None
    with pytest.raises(ValueError):
        msim.result_keys()
    msim.base_sim = msim.sims[0] # Restore

    # Run
    msim.run(verbose=verbose)
    msim.reduce(quantiles=[0.1, 0.9], output=True)
    with pytest.raises(ValueError):
        msim.reduce(quantiles='invalid')
    msim.compare(output=True, do_plot=True, log_scale=False)

    # Plot
    for i in range(2):
        if i == 1:
            msim.reset() # Reset as if reduce() was not called
        msim.plot()
        msim.plot_result('r_eff')
    print('↑ May print some plotting warnings')

    # Save
    for keep_people in [True, False]:
        msim.save(filename=msim_path, keep_people=keep_people)

    # Scenarios
    scens = cv.Scenarios(sim=s1, metapars={'n_runs':1})
    scens.run(keep_people=True, verbose=verbose, debug=debug)
    for keep_people in [True, False]:
        scens.save(scens_path, keep_people=keep_people)
    cv.Scenarios.load(scens_path)

    # Tidy up
    remove_files(msim_path, scens_path)

    return


def test_sim():
    sc.heading('Testing sim')

    # Test resetting layer parameters
    sim = cv.Sim(pop_size=100, label='test_label')
    sim.reset_layer_pars()
    sim.initialize()
    sim.reset_layer_pars()

    # Test validation
    sim['pop_size'] = 'invalid'
    with pytest.raises(ValueError):
        sim.validate_pars()
    sim['pop_size'] = 100 # Restore

    # Handle missing start day
    sim['start_day'] = None
    sim.validate_pars()

    # Can't have an end day before the start day
    sim['end_day'] = '2019-01-01'
    with pytest.raises(ValueError):
        sim.validate_pars()

    # Can't have both end_days and n_days None
    sim['end_day'] = None
    sim['n_days'] = None
    with pytest.raises(ValueError):
        sim.validate_pars()
    sim['n_days'] = 30 # Restore

    # Check layer pars are internally consistent
    sim['quar_factor'] = {'invalid':30}
    with pytest.raises(sc.KeyNotFoundError):
        sim.validate_pars()
    sim.reset_layer_pars() # Restore

    # Check mismatch with population
    for key in ['beta_layer', 'contacts', 'quar_factor']:
        sim[key] = {'invalid':1}
    with pytest.raises(sc.KeyNotFoundError):
        sim.validate_pars()
    sim.reset_layer_pars() # Restore

    # Convert interventions dict to intervention
    sim['interventions'] = {'which': 'change_beta', 'pars': {'days': 10, 'changes': 0.5}}
    sim.validate_pars()

    # Test intervention functions and results analyses
    cv.Sim(pop_size=100, verbose=0, interventions=lambda sim: (sim.t==20 and (sim.__setitem__('beta', 0) or print(f'Applying lambda intervention to set beta=0 on day {sim.t}')))).run() # ...This is not the recommended way of defining interventions.

    # Test other outputs
    sim = cv.Sim(pop_size=100, verbose=0, n_days=30)
    sim.run()
    sim.compute_r_eff(method='infectious')
    sim.compute_r_eff(method='outcome')
    sim.compute_gen_time()

    # Plot results
    sim.plot_result('r_eff')

    return


#%% Run as a script
if __name__ == '__main__':

    # We need to create plots to test plotting, but can use a non-GUI backend
    if not do_plot:
        pl.switch_backend('agg')

    T = sc.tic()

    test_base()
    test_interventions()
    test_misc()
    test_people()
    test_plotting()
    test_population()
    test_requirements()
    test_run()
    test_sim()

    print('\n'*2)
    sc.toc(T)
    print('Done.')