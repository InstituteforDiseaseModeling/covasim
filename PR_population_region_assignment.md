# Add Configurable Population-Based Region Assignment

## Summary

This PR adds support for assigning agents to configurable regions or sewersheds using population data loaded from a CSV file.

A new `pop_code` parameter selects the population dataset to load. Each agent is then assigned a `region_code` according to the population distribution defined in that dataset.

This provides a flexible mechanism that can be used for ZIP codes, sewersheds, grid cells, administrative regions, or other population-based geographic groupings.

## Changes

### `parameters.py`

Added a new population data configuration parameter:

```python
pars['pop_code'] = 'test'  # Population code used to load population data
```

The default value is `test`, which points to a small population dataset intended for testing.

### `population.py`

Added `assign_regions(pars)`:

```python
def assign_regions(pars):
    """Assign each agent to a configurable region or sewershed."""
    pop_size = int(pars['pop_size'])

    population_data = cvdata.get_population_data(pars['pop_code'])
    if population_data is None:
        errormsg = f'Could not load population data for requested location "{pars["location"]}"'
        raise ValueError(errormsg)

    labels = population_data['region_code'].values
    probabilities = population_data['probability'].values
    probabilities = probabilities / np.sum(probabilities)

    return np.random.choice(
        labels,
        size=pop_size,
        p=probabilities
    ).astype(cvd.default_int)
```

Region assignment is used as:

```python
regions = assign_regions(pars)
```

Each agent receives a region according to the relative population of each region in the selected population dataset.

### `loaders.py`

Added `get_population_data()` to load population data from CSV:

```python
def get_population_data(code='test'):
    """
    Function to load population data from a CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing population data.

    Example data format:
        region_code,population
        1,1000000
        2,500000
        3,750000
        4,250000
    """
    try:
        population_data = pd.read_csv(f'data/population/{code}.csv')
        population_data['probability'] = (
            population_data['population']
            / population_data['population'].sum()
        )
        return population_data

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Population data file not found for code: {code}"
        )
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while loading population data: {e}"
        )
```

The loader:

1. Reads `data/population/<pop_code>.csv`.
2. Calculates the total population.
3. Converts each region's population into a sampling probability.
4. Returns the population data for region assignment.

## Test Population Data

Added:

```text
data/population/test.csv
```

Contents:

```csv
region_code,population
1,1000000
2,500000
3,750000
4,250000
```

The corresponding sampling probabilities are based on each region's share of the total population.

## Usage

Select the population dataset through:

```python
pars['pop_code'] = 'test'
```

For example, additional datasets could be added as:

```text
data/population/atlanta_zip_codes.csv
data/population/fulton_sewersheds.csv
data/population/seattle_grid_cells.csv
```

and selected with:

```python
pars['pop_code'] = 'atlanta_zip_codes'
```

As long as the CSV contains:

```csv
region_code,population
```

the same assignment logic can be reused.

## Motivation

This change makes geographic assignment configurable instead of hard-coded. It allows the population model to represent different geographic resolutions or region definitions without changing the core population generation logic.

Potential use cases include:

- ZIP-code-based populations
- Sewershed assignment
- Grid-based spatial models
- Administrative regions
- Custom geographic groupings
