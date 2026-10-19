## Sewershed Assignment Implementation

The current sewershed assignment workflow is:

1. Regions are assigned to the population.
2. Coordinates are generated for each person based on their assigned region.
3. Sewersheds are assigned based on those coordinates.

`shapely` has also been added to the project dependencies for polygon handling and point-in-polygon operations.

### Sample Sewershed File

A sample `sewersheds.csv` file was added. For testing, it divides the entire world into four sewersheds:

```csv
sid,polygon
1,"[[-180,-90],[0,-90],[0,0],[-180,0],[-180,-90]]"
2,"[[0,-90],[180,-90],[180,0],[0,0],[0,-90]]"
3,"[[-180,0],[0,0],[0,90],[-180,90],[-180,0]]"
4,"[[0,0],[180,0],[180,90],[0,90],[0,0]]"
```

### Changes

#### `covasim/defaults.py`

The following properties were added for each person:

```python
'sewershed',  # Int: 0 default catchment, -1 outside supplied polygons
'x',          # Float64 home coordinate
'y',          # Float64 home coordinate
```

#### `covasim/parameters.py`

The following parameters were added:

```python
pars['sewershed_file'] = None  # Optional CSV with sid,polygon; None means one whole-population catchment
pars['people_coords'] = None   # Optional (pop_size, 2) home coordinates, in polygon coordinate system
```

#### `covasim/people.py`

Initialization for the new person properties was added:

```python
elif key == 'sewershed':
    self[key] = np.zeros(self.pars['pop_size'], dtype=cvd.default_int)
elif key in ['x', 'y']:
    self[key] = np.full(self.pars['pop_size'], np.nan, dtype=np.float64)
```

### Current Naive Implementation

A naive implementation has been completed using four regions that divide the entire world.

The four polygons are:

```text
{
    1: <POLYGON ((-180 -90, 0 -90, 0 0, -180 0, -180 -90))>,
    2: <POLYGON ((0 -90, 180 -90, 180 0, 0 0, 0 -90))>,
    3: <POLYGON ((-180 0, 0 0, 0 90, -180 90, -180 0))>,
    4: <POLYGON ((0 0, 180 0, 180 90, 0 90, 0 0))>
}
```

An example sewershed assignment for 20 people is:

```text
[4 2 3 1 2 1 3 4 4 2 2 1 2 3 3 3 3 1 2 1]
```

### Running the Implementation

The implementation can currently be tested with:

```python
import covasim as cv

sim = cv.Sim(
    pop_size=20,
    pop_infected=0,
    sewershed_file='sewersheds.csv',
)

people = cv.make_people(sim)
print(people.sewershed)
```

### TODO

The main remaining issue is coordinate generation for regions.

Currently, the minimum and maximum coordinates used to generate person locations are hardcoded to cover the entire world. These bounds should instead be:

- Read from region boundary data, or
- Derived from the available population/region data.

This needs to be updated so that generated coordinates reflect the actual geographic boundaries of each region, which will allow accurate sewershed assignment.
