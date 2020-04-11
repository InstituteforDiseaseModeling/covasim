'''
This script creates a single file containing all the scraped 
data from the Corona Data Scraper.
'''

import os
import sys
import logging
import pandas as pd
import sciris as sc

subfolder = 'epi_data'
outputfile = 'corona_data_scraper.csv'

log = logging.getLogger("Corona Data Scraper Data Loader")
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# Read in the Corona Data Scraper Data into a dataframe.
log.info("Loading timeseries data from coronadatascraper.com")
cds = pd.read_csv("https://coronadatascraper.com/timeseries.csv")
log.info(f"Loaded {len(cds)} records; now transforming data")

# Just to be safe, let's sort by name and date. Probably not 
# necessary but better safe than sorry!.
cds = cds.sort_values(['name', 'date'])

# Each data set has a unique name. Let's create groups.
g = cds.groupby('name')

# The parameter 'day' is the number of days since the first 
# day of data collection for the group.
cds['day'] = g['date'].transform(lambda x: (pd.to_datetime(
    x) - min(pd.to_datetime(x)))).apply(lambda x: x.days)

# We'll 'rename' some of the columns to be consistent
# with the parameters file.
cds['positives'] = cds.cases
cds['death'] = cds.deaths
cds['tests'] = cds.tested
cds.drop(['cases', 'deaths', 'tested'], inplace=True, axis=1)

# The Corona Data Scraper Data contains cumulative totals
# for each group. We want _new_ amounts per the previous
# (reporting) day. So we 'shift' or lag the data by
# one.
cds["positives_lagged"] = g.positives.shift(1)
cds["death_lagged"] = g.death.shift(1)
cds['active_lagged'] = g.active.shift(1)
cds['tests_lagged'] = g.tests.shift(1)

# And subtract the current number from the previous reported
# day, filling in the first number with data from the first
# row of the group. Other N/As will remain N/A.
cds['new_positives'] = cds.positives.sub(
    cds.positives_lagged).fillna(cds.positives)
cds['new_active'] = cds.active.sub(cds.active_lagged).fillna(cds.active)
cds['new_death'] = cds.death.sub(cds.death_lagged).fillna(cds.death)
cds['new_tests'] = cds.tests.sub(cds.tests_lagged).fillna(cds.tests)

# We'll drop the unneeded lag data.
cds.drop(['positives_lagged', 'death_lagged', 'active_lagged', 'tests_lagged'], inplace=True, axis=1)

# And save it to the data directory.
here = sc.thisdir(__file__)
data_home = os.path.join(here, subfolder)
filepath = sc.makefilepath(filename=outputfile, folder=data_home)
log.info(f"Saving to {filepath}")
cds.to_csv(filepath)
log.info(f"Script complete")
