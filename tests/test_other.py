'''
Tests for things that do not belong in other files.
'''

#%% Imports and settings
import os
import pytest
import sciris as sc
import covasim as cv
import pylab as pl

do_plot = False


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
    for path in [json_path, sim_path]:
        print(f'Removing {path}')
        os.remove(path)

    return


def test_requirements():
    sc.heading('Testing requirements')

    return


#%% Run as a script
if __name__ == '__main__':

    if not do_plot:
        pl.switch_backend('agg')

    sc.tic()

    test_base()
    test_requirements()

    print('\n'*2)
    sc.toc()