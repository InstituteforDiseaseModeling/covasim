'''
Test the rank-based population type (pop_type='rank'). Needs no raster or optional packages.
'''
import os
import numpy as np
import pytest
import covasim as cv

sd_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'tutorials', 'san-diego-data') # Example boundaries and sewersheds


def test_rank_population():
    ''' pop_type='rank' should give a single static contact layer and run '''
    pars = cv.make_pars(pop_type='rank')
    assert pars['contacts'] == {'a': 20}
    assert pars['dynam_layer'] == {'a': 0} # Static, so the rank structure is never redrawn

    sim = cv.Sim(pop_type='rank', pop_size=20000, n_days=10, verbose=0).run()
    assert list(sim.people.contacts.keys()) == ['a']
    assert len(sim.people.contacts['a']) > 0 # The layer has contacts


def test_rank_population_workers():
    ''' use_workers=True should build the same kind of layer. It needs at least 4,000 agents to use more than one worker,
    and stays on one core if the computer cannot run workers, so it passes on any machine '''
    sim = cv.Sim(pop_type='rank', pop_size=20000, n_days=5, verbose=0)
    sim.initialize(use_workers=True)
    sim.run()
    assert list(sim.people.contacts.keys()) == ['a']
    assert len(sim.people.contacts['a']) > 0


def test_rank_population_uses_boundary_coords():
    ''' With a boundary file, rank contacts must be built from the same locations as people.x/y and the sewersheds '''
    sim = cv.Sim(pop_type='rank', pop_size=4000, n_days=2, verbose=0,
                 boundary_shapefile=os.path.join(sd_dir, 'boundaries.geojson'),
                 sewershed_file=os.path.join(sd_dir, 'sewersheds.geojson'))
    sim.initialize()
    p = sim.people
    assert -118 < p.x.min() and p.x.max() < -116 and 32 < p.y.min() and p.y.max() < 34 # lon/lat, not km
    assert set(np.unique(p.sewershed)) <= {-1, 1, 2, 3}

    # Contacts should be spatially local: far shorter than between randomly chosen pairs
    c = p.contacts['a']
    contact_dist = np.hypot(p.x[c['p1']] - p.x[c['p2']], p.y[c['p1']] - p.y[c['p2']]).mean()
    perm = np.random.permutation(len(p.x))
    random_dist = np.hypot(p.x - p.x[perm], p.y - p.y[perm]).mean()
    assert contact_dist < 0.5*random_dist


def test_rank_population_density_within_boundaries(tmp_path):
    ''' With a raster and boundaries, people follow the raster density but stay inside the boundary polygons '''
    pytest.importorskip('rasterio')
    import rasterio
    from rasterio.transform import from_origin

    # A synthetic raster covering more than the boundaries, with all the people in its western half
    lon0, lat0, size, nx, ny = -117.4, 33.0, 0.01, 40, 40 # lon -117.4 to -117.0, lat 32.6 to 33.0
    counts = np.ones((ny, nx), dtype='float32')
    counts[:, nx//2:] = 0
    path = str(tmp_path/'density.tif')
    with rasterio.open(path, 'w', driver='GTiff', height=ny, width=nx, count=1, dtype='float32', crs='EPSG:4326', transform=from_origin(lon0, lat0, size, size)) as dst:
        dst.write(counts, 1)

    sim = cv.Sim(pop_type='rank', pop_size=3000, n_days=2, verbose=0, density_raster=path,
                 boundary_shapefile=os.path.join(sd_dir, 'boundaries.geojson'),
                 sewershed_file=os.path.join(sd_dir, 'sewersheds.geojson'))
    sim.initialize()
    p = sim.people
    assert p.x.max() < -117.2 # Only the western half of the raster has people
    import geopandas as gpd
    boundary = gpd.read_file(os.path.join(sd_dir, 'boundaries.geojson')).union_all()
    import shapely
    assert shapely.contains_xy(boundary, p.x, p.y).all() # Everyone inside the boundary polygons
    assert len(np.unique(p.region)) >= 1 and (p.sewershed >= 0).any()
    c = p.contacts['a']
    contact_dist = np.hypot(p.x[c['p1']] - p.x[c['p2']], p.y[c['p1']] - p.y[c['p2']]).mean()
    perm = np.random.permutation(len(p.x))
    assert contact_dist < 0.5*np.hypot(p.x - p.x[perm], p.y - p.y[perm]).mean()


if __name__ == '__main__':
    test_rank_population()
    test_rank_population_workers()