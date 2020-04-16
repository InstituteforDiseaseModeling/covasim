# Data scrapers

These scripts pull data from various sources for use in Covasim. To run all scrapers,
simply type

```bash
./run_scrapers 
```

## 1. Corona Data Scraper

To quote the [Corona Data Scraper](https://coronadatascraper.com) web page,

> Corona Data Scraper pulls COVID-19 Coronavirus case data from verified sources, finds the corresponding GeoJSON features, and adds population data.

We transform this data for use in the Covasim parameter format. It is stored
in CSV-format. 


### Updating

To update the  Corona Data Scraper data,

```bash
python data/load_corona_data_scraper_data.py 
```

This will create a file `corona_data_scraper.csv` in the `data/epi_data` directory.


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

As of April 4, 2020, there are apparently 3280 data sets.



## 2. European Centre for Disease Prevention and Control 

To quote the [European Centre for Disease Prevention and Control ](https://www.ecdc.europa.eu/en/geographical-distribution-2019-ncov-cases) web page,

> Since the beginning of the coronavirus pandemic, ECDC’s Epidemic Intelligence team has been collecting the number of COVID-19 cases and deaths, based on reports from health authorities worldwide. This comprehensive and systematic process is carried out on a daily basis. To insure the accuracy and reliability of the data, this process is being constantly refined. This helps to monitor and interpret the dynamics of the COVID-19 pandemic not only in the European Union (EU), the European Economic Area (EEA), but also worldwide.

We transform this data for use in the Covasim parameter format. It is stored
in CSV-format. 


### Updating

To update the Corona Data Scraper data,

```bash
python data/load_ecdp_data.py 
```

This will create a file `ecdp_data.csv` in the `data/epi_data` directory.

This adds data from 204 countries and territories (as of April 5, 2020), including Africa, Asia, the Americas, Europe, and Oceania. More details at: https://www.ecdc.europa.eu/en/geographical-distribution-2019-ncov-cases

The following columns are present in the data:

- `countriesAndTerritories`: Unique country or territory name
- `geoId`: Geo ID of same
- `countryterritoryCode`: ISO 3-letter code?
- `date`: Date in yyyy-MM-dd text format
- `day`: Number of days since first reporting
- `new_positives`: New positives on this date
- `new_death`: Number of deaths on this date



## 3. The COVID Tracking Project

The COVID Tracking Project "obtains, organizes, and publishes high-quality data required to understand and respond to the COVID-19 outbreak in the United States." The project website is https://covidtracking.com

We transform this data for use in the Covasim parameter format. It is stored
in CSV-format. 


### Updating

To update the COVID Tracking Project data,

```bash
python data/load_covid_tracking_project_data.py
```

This will create a file `covid-tracking-project-data.csv` in the data directory.

This adds data from each of the US states and territories, as well as for the whole of the United States. The following fields are saved:

- `date`
- `positive`
- `negative`
- `pending`
- `hospitalizedCurrently`
- `hospitalizedCumulative`
- `inIcuCurrently`
- `inIcuCumulative`
- `onVentilatorCurrently`
- `onVentilatorCumulative`
- `recovered`
- `hash`
- `dateChecked`
- `death`
- `hospitalized`
- `total`
- `totalTestResults`
- `posNeg`
- `fips`
- `hospitalizedIncrease`
- `negativeIncrease`
- `name`
- `day`
- `new_positives`
- `new_negatives`
- `new_tests`
- `new_death`
- `new_icu`
- `new_vent`

More details at: https://covidtracking.com/api The `new_` variables are per-day
changes in the values, in parameter.py format.


## 4. Demographic data scraper

To scrape demographic data, run

```bash
python data/load_demographic_data.py
```