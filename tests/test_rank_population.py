'''
Test the rank-based population type (pop_type='rank'). Needs no raster or optional packages.
'''
import covasim as cv


def test_rank_population():
    ''' pop_type='rank' should give a single static contact layer and run '''
    pars = cv.make_pars(pop_type='rank')
    assert pars['contacts'] == {'a': 20}
    assert pars['dynam_layer'] == {'a': 0} # Static, so the rank structure is never redrawn

    sim = cv.Sim(pop_type='rank', pop_size=500, n_days=10, verbose=0).run()
    assert list(sim.people.contacts.keys()) == ['a']
    assert len(sim.people.contacts['a']) > 0 # The layer has contacts


if __name__ == '__main__':
    test_rank_population()