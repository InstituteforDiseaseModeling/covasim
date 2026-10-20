
'''
Geographic helpers: place agents on a map using a population-density raster.

Nothing in this file is specific to any one place. The caller supplies a raster of
population counts (a GeoTIFF in longitude/latitude, for example from WorldPop), and
these functions turn it into agent coordinates in kilometres, which can then be used
for location-aware contact networks such as make_rank_based_contacts().

Functions:
    load_density_raster(path, bbox)
        Read the raster file (a GeoTIFF) into a grid of people per cell. bbox optionally
        reads only a window of the raster instead of the whole file.
    sample_coords_from_density(raster, pop_size)
        Take the loaded raster and place the agents: pick a raster cell for each agent with
        probability proportional to its population, then a random spot inside that cell.
        Returns (longitude, latitude) in degrees.
    lonlat_to_km(lonlat)
        Convert the (longitude, latitude) points sampled from the raster into a flat map in
        km (x, y), so that straight-line distance is a real distance. Accurate for areas up
        to about county size.
    assign_coords(pars)
        The entry point. Reads the raster path (density_raster) and the optional window
        (density_bbox) from pars, then loads the raster, samples agent locations from it and
        converts them to km. Returns an (N, 2) array of x, y in km.

How it is used (pop_type='rank'):
    1. The raster is passed in as a Sim parameter, so a different raster places the agents
       somewhere else:
           sim = cv.Sim(pop_type='rank', pop_size=N, density_raster='path/to/raster.tif')
       To read only part of a large raster, also pass
           density_bbox=(lon_min, lat_min, lon_max, lat_max)
       Both parameters default to None in make_pars().
    2. make_randpop() calls assign_density_based_coords(pars) and passes the resulting x, y array to
       make_rank_based_contacts().
    If no raster is given, agents are spread uniformly over a 10 x 10 km square, which is
    only useful for testing.

With a boundary file (boundary_shapefile) as well, agents are still placed according to the raster, but only inside
the boundary polygons (sample_coords_within_boundaries()); their region is the polygon they land in
(assign_regions_from_coords()) and the region 'population' column is not used. Locations then stay in longitude/latitude,
so they can be matched to sewershed polygons; only the contact network converts them to km.

The Sim's 'location' parameter (which loads national age and household data) is not used
here; only density_raster and density_bbox decide where agents are placed.

The raster library (rasterio) is an optional dependency, only needed by
load_density_raster(): pip install rasterio

'''

import numpy as np

__all__ = ['load_density_raster', 'sample_coords_from_density', 'sample_coords_within_boundaries', 'assign_regions_from_coords',
           'lonlat_to_km', 'assign_density_based_coords']


def load_density_raster(path, bbox=None):
    '''
    Read a population-count GeoTIFF (people per pixel) in longitude/latitude.

    Args:
        path (str): path to the .tif
        bbox (tuple): optional (lon_min, lat_min, lon_max, lat_max) window to read; None reads the whole file

    Returns:
        dict with 'counts' (2D array of people per pixel), 'lons' (1D pixel-centre longitudes),
        'lats' (1D pixel-centre latitudes), 'dlon', 'dlat' (pixel size in degrees)
    '''
    try:
        import rasterio # Optional dependency
        from rasterio.windows import from_bounds
    except ModuleNotFoundError as E: # pragma: no cover
        errormsg = 'Reading a density raster needs the optional rasterio package: pip install rasterio'
        raise ModuleNotFoundError(errormsg) from E

    with rasterio.open(path) as src:
        if src.crs is not None and not src.crs.is_geographic: # The coordinate conversion below assumes degrees
            errormsg = f'The raster must use longitude/latitude (e.g. EPSG:4326), but its CRS is {src.crs}'
            raise ValueError(errormsg)
        if bbox is not None:
            window = from_bounds(*bbox, transform=src.transform)
            window = window.round_offsets().round_lengths()
        else:
            window = rasterio.windows.Window(0, 0, src.width, src.height)
        counts = src.read(1, window=window).astype(float)
        transform = src.window_transform(window)
        nodata = src.nodata

    if nodata is not None:
        counts[counts == nodata] = 0
    counts[~np.isfinite(counts)] = 0
    counts[counts < 0] = 0

    nrows, ncols = counts.shape
    dlon, dlat = transform.a, transform.e # dlat is negative for north-up rasters
    lons = transform.c + dlon*(np.arange(ncols) + 0.5)
    lats = transform.f + dlat*(np.arange(nrows) + 0.5)
    return dict(counts=counts, lons=lons, lats=lats, dlon=dlon, dlat=dlat)


def sample_coords_from_density(raster, pop_size):
    '''
    Draw pop_size (lon, lat) points: pick a pixel with probability proportional
    to its population count, then place the agent uniformly within the pixel.

    Args:
        raster   (dict): as returned by load_density_raster()
        pop_size (int):  number of agents to place

    Returns:
        (pop_size, 2) array of longitude, latitude in degrees
    '''
    counts = raster['counts']
    probs = counts.ravel()/counts.sum()
    pix = np.random.choice(probs.size, size=pop_size, p=probs)
    rows, cols = np.unravel_index(pix, counts.shape)
    lon = raster['lons'][cols] + (np.random.random(pop_size) - 0.5)*abs(raster['dlon'])
    lat = raster['lats'][rows] + (np.random.random(pop_size) - 0.5)*abs(raster['dlat'])
    return np.column_stack((lon, lat))


def sample_coords_within_boundaries(raster, boundaries, pop_size, max_tries=100):
    '''
    Draw pop_size (lon, lat) points from a population-density raster, but only inside the boundary polygons.

    Pixels whose centre is outside every polygon get zero weight; the rest are chosen in proportion to
    their population, and the agent is placed at a random spot in the pixel. A spot that falls just outside
    the boundary (pixels on the edge) is redrawn until it is inside.

    Args:
        raster     (dict):         as returned by load_density_raster(), in longitude/latitude
        boundaries (GeoDataFrame): region polygons in longitude/latitude (EPSG:4326)
        pop_size   (int):          number of agents to place
        max_tries  (int):          maximum number of redraws for agents that land outside the boundary

    Returns:
        (pop_size, 2) array of longitude, latitude in degrees
    '''
    import shapely # Imported here since only the boundary-based placement needs it

    boundary = shapely.union_all(boundaries.geometry.values) # All regions as one shape
    lon_grid, lat_grid = np.meshgrid(raster['lons'], raster['lats'])
    inside = shapely.contains_xy(boundary, lon_grid.ravel(), lat_grid.ravel()).reshape(lon_grid.shape)
    masked = dict(raster, counts=raster['counts']*inside) # People per pixel, zero outside the boundary
    if masked['counts'].sum() <= 0:
        errormsg = 'The density raster has no population inside the boundary polygons; check that they overlap and use the same coordinates (longitude/latitude)'
        raise ValueError(errormsg)

    pop_size = int(pop_size)
    coords = sample_coords_from_density(masked, pop_size)
    for _ in range(max_tries):
        outside = ~shapely.contains_xy(boundary, coords[:, 0], coords[:, 1])
        if not outside.any():
            return coords
        coords[outside] = sample_coords_from_density(masked, int(outside.sum())) # Redraw only those outside
    errormsg = f'Could not place all agents inside the boundary polygons after {max_tries} tries'
    raise RuntimeError(errormsg)


def assign_regions_from_coords(lonlat, boundaries):
    '''
    Give each agent the region_code of the boundary polygon that contains its location.
    Where polygons overlap, the first one in the file wins.

    Args:
        lonlat     (array):        (N, 2) array of longitude, latitude in degrees
        boundaries (GeoDataFrame): region polygons (EPSG:4326) with a region_code column

    Returns:
        array of region_code values, one per agent
    '''
    import shapely
    labels = boundaries['region_code'].to_numpy()
    index  = np.full(len(lonlat), -1)
    for i, geom in enumerate(boundaries.geometry.values):
        unassigned = index < 0
        hit = shapely.contains_xy(geom, lonlat[unassigned, 0], lonlat[unassigned, 1])
        index[np.nonzero(unassigned)[0][hit]] = i
    if (index < 0).any():
        errormsg = f'{int((index < 0).sum())} agents are outside all boundary polygons'
        raise ValueError(errormsg)
    return labels[index]


def lonlat_to_km(lonlat):
    '''
    Convert (lon, lat) in degrees to (x, y) in km, using an equirectangular projection
    about the mean latitude. This is accurate for areas of city to county size, so that
    straight-line distance between two rows is a good distance in km. It is not suitable
    for areas spanning many degrees of latitude or reaching the poles.

    Args:
        lonlat (array): (N, 2) array of longitude, latitude in degrees

    Returns:
        (N, 2) array of x, y in km, centred on the mean location
    '''
    lon0, lat0 = lonlat[:, 0].mean(), lonlat[:, 1].mean()
    kx = 111.320*np.cos(np.radians(lat0)) # km per degree of longitude at this latitude
    ky = 110.574 # km per degree of latitude
    return np.column_stack(((lonlat[:, 0]-lon0)*kx, (lonlat[:, 1]-lat0)*ky))


def _get_par(pars, key, default=None):
    ''' Read a parameter from either a dict or a Sim (which supports sim[key] but not .get()) '''
    try:
        return pars[key]
    except (KeyError, AttributeError): # sciris' KeyNotFoundError is a KeyError
        return default

def assign_density_based_coords(pars):
    '''
    Assign each agent an (x, y) location in km, sampled from a population-density raster.

    Steps:
        1. Read the raster path and optional window from pars.
        2. Load the raster (a grid of people per cell).
        3. Sample a (longitude, latitude) for each agent from the raster.
        4. Convert those degrees to km.

    Args:
        pars (dict or Sim): needs 'pop_size'; optionally 'density_raster' (path to the .tif)
                            and 'density_bbox' (lon_min, lat_min, lon_max, lat_max)

    Returns:
        (N, 2) array of x, y in km. If no raster is given, agents are spread uniformly over
        a 10 x 10 km square, which is only useful for testing.
    '''

    # 1. Read the inputs
    pop_size = int(pars['pop_size'])                  # Number of agents to place
    path     = _get_par(pars, 'density_raster')       # Path to the raster file (None if not supplied)
    bbox     = _get_par(pars, 'density_bbox')         # Optional (lon_min, lat_min, lon_max, lat_max) window

    # No raster: fall back to a 10 x 10 km test square
    if path is None:
        return np.random.uniform(0, 10, size=(pop_size, 2))

    # 2. Load the raster: a grid of people per cell, plus the longitude/latitude of each cell
    raster = load_density_raster(path, bbox)

    # 3. Sample a (longitude, latitude) for each agent: a cell chosen in proportion to its population,
    #    then a random spot inside that cell
    lonlat = sample_coords_from_density(raster, pop_size)

    # 4. Convert degrees to km, so that straight-line distance between agents is a real distance
    xy = lonlat_to_km(lonlat)

    return xy


#********************
#Additional
 
def lonlat_to_xyz_km(lonlat, radius=6371.0):
    '''
    Convert (lon, lat) in degrees to 3D points (x, y, z) in km on a sphere the size of the Earth.
 
    The straight-line (chord) distance between two such points grows with the distance
    along the surface, so which points are nearest, second nearest, and so on is exactly the
    same as with great-circle distances, at any scale and anywhere on the globe. Use these
    as the coordinates for rank-based (nearest-neighbour) calculations. Chord lengths are
    very slightly shorter than surface distances (0.04% at 1,000 km); for rankings this
    makes no difference. The Earth is treated as a sphere, ignoring its slight flattening.
 
    Args:
        lonlat (array): (N, 2) array of longitude, latitude in degrees
        radius (float): sphere radius in km (default: mean Earth radius)
 
    Returns:
        (N, 3) array of x, y, z in km
    '''
    lon = np.radians(lonlat[:, 0])
    lat = np.radians(lonlat[:, 1])
    return radius*np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
 