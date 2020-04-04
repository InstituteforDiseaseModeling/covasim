# Data Loaders

## 1. Corona Data Scraper


To quote the [Corona Data Scraper](https://coronadatascraper.com) web page,

> Corona Data Scraper pulls COVID-19 Coronavirus case data from verified sources, finds the corresponding GeoJSON features, and adds population data.

We transform this data for use in the Covasim parameter format. It is stored
in CSV-format. 

### Updating

To update the  Corona Data Scraper data,

```bash
python data_loaders/load_corona_data_scraper_data.py 
```

This will create a file `corona_data_scraper.csv` in the data directory.

### Data dictionary

The following columns are present in the data:

- `name`: Apparently, a unique name for the data set
- `level`: level of the data (country, state, county, city)
- `city`: The city name
- `county`: The county or parish
- `state`: The state, province, or region
- `country`: [ ISO 3166-1 alpha-3 country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3)
- `population`: The estimated population of the location
- `lat`: latitude
- `long`: longitude
- `url`: Data source
- aggregate
- `tz`: Time zone
- `recovered`: Cumulative number recovered (as of date)
- `active`: Cumulative number active (as of date)
- `growthFactor`:
- `date`: Date in yyyy-MM-dd text format
- `day`: Number of days since first reporting
- `positives`: Cumulative number of positive cases (as of date)
- `death`: Cumulative number of deaths (as of date)
- `tests`: Cumulative number of tests (as of date)
- `new_positives`: New positives on this date
- `new_active`: New active cases on this date
- `new_death`: Number of deaths on this date
- `new_tests`: New tests on this date

As of April 4, 2020, There are apparently 3280 data sets.
