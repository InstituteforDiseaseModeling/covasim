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

do_plot = False

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
    sim = cv.Sim(pop_size=100, verbose=0)
    sim.run()

    # Check setting invalid key
    with pytest.raises(sc.KeyNotFoundError):
        po = cv.ParsObj(pars={'a':2, 'b':3})
        po.update_pars({'c':4})

    # Printing result
    r = cv.Result()
    print(r)
    print(r.npts)

    # Day conversion
    daystr = '2020-04-04'
    sim.day(daystr)
    sim.day(sc.readdate(daystr))
    with pytest.raises(ValueError):
        sim.day('not a date')

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
    ppl.keys(which='all_states')
    ppl.index()
    ppl.resize(pop_size=200)
    ppl.to_df()
    ppl.to_arr()
    ppl.person(50)
    people = ppl.to_people()
    ppl.from_people(people)
    with pytest.raises(sc.KeyNotFoundError):
        ppl.make_edgelist([{'invalid_key':[0,1,2]}])

    # Contacts methods
    contacts = ppl.contacts
    df = contacts['a'].to_df()
    ppl.remove_duplicates(df)
    with pytest.raises(sc.KeyNotFoundError):
        contacts['invalid_key']
    contacts.values()
    len(contacts)

    # Transmission tree methods
    ppl.transtree.make_targets()
    ppl.make_detailed_transtree()
    ppl.transtree.plot()
    ppl.transtree.animate(animate=False)

    # Tidy up
    remove_files(json_path, sim_path)

    return


def test_misc():
    sc.heading('Testing miscellaneous functions')

    sim_path = 'test_misc.sim'
    json_path = 'test_misc.json'

    # Data loading
    cv.load_data('example_data.csv')
    cv.load_data('example_data.xlsx')

    with pytest.raises(NotImplementedError):
        cv.load_data('example_data.unsupported_extension')

    with pytest.raises(ValueError):
        cv.load_data('example_data.xlsx', columns=['missing_column'])

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
    sc.heading('Testing people (dynamic layers)')

    sim = cv.Sim(pop_size=100, n_days=10, verbose=0, dynam_layer={'a':1})
    sim.run()

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
        sim = cv.Sim(pop_size=5000, pop_type='synthpops')
        sim.initialize()
    except Exception as E:
        errormsg = f'Synthpops test did not pass:\n{str(E)}\nNote: synthpops is optional so this exception is OK.'
        print(errormsg)

    # Not working
    with pytest.raises(ValueError):
        sim = cv.Sim(pop_type='not_an_option')
        sim.initialize()

    # Save/load
    sim = cv.Sim(pop_size=100, popfile=pop_path)
    sim.initialize(save_pop=True)
    cv.Sim(pop_size=100, popfile=pop_path, load_pop=True)
    with pytest.raises(ValueError):
        cv.Sim(pop_size=101, popfile=pop_path, load_pop=True)

    remove_files(pop_path)

    return



def test_requirements():
    sc.heading('Testing requirements')

    with pytest.raises(ImportError):
        cv.requirements.min_versions['sciris'] = '99.99.99'
        cv.requirements.check_sciris()

    with pytest.raises(ImportError):
        cv.requirements.min_versions['scirisweb'] = '99.99.99'
        cv.requirements.check_scirisweb(die=True)

    cv.requirements.check_synthpops()

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
    msim.run(verbose=0)
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

    # Save
    for keep_people in [True, False]:
        msim.save(filename=msim_path, keep_people=keep_people)

    # Scenarios
    scens = cv.Scenarios(sim=s1, metapars={'n_runs':1})
    scens.run(keep_people=True, verbose=0)
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
    sim['quar_eff'] = {'invalid':30}
    with pytest.raises(sc.KeyNotFoundError):
        sim.validate_pars()
    sim.reset_layer_pars() # Restore

    # Check mismatch with population
    for key in ['beta_layer', 'contacts', 'quar_eff']:
        sim[key] = {'invalid':1}
    with pytest.raises(sc.KeyNotFoundError):
        sim.validate_pars()
    sim.reset_layer_pars() # Restore

    # Convert interventions dict to intervention
    sim['interventions'] = {'which': 'change_beta', 'pars': {'days': 10, 'changes': 0.5}}
    sim.validate_pars()

    # Test intervention functions and results analyses
    sim = cv.Sim(pop_size=100)
    sim['interv_func'] = lambda sim: (sim.t==20 and (sim.__setitem__('beta', 0) or print(f'Applying lambda intervention to set beta=0 on day {sim.t}'))) # The world's most ridiculous way of defining an intervention
    sim['verbose'] = 0
    sim.run()
    sim.compute_r_eff(method='infectious')
    sim.compute_r_eff(method='outcome')
    sim.compute_gen_time()

    # Plot results
    sim.plot_result('r_eff')

    return


#%% Run as a script
if __name__ == '__main__':

    if not do_plot:
        pl.switch_backend('agg')

    sc.tic()

    test_base()
    test_misc()
    test_people()
    test_population()
    test_requirements()
    test_sim()
    test_run()

    print('\n'*2)
    sc.toc()