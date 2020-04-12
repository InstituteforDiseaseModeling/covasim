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
df = pd.read_csv("https://coronadatascraper.com/timeseries.csv")
log.info(f"Loaded {len(df)} records; now transforming data")

# We'll rename some of the columns to be consistent
# with the parameters file.
# Add change the 'name' field to 'key'
df = df.rename(columns={'name': 'key', 'cases': 'positives',
                        'deaths': 'death', 'tested': 'tests'})


# Just to be safe, let's sort by name and date. Probably not 
# necessary but better safe than sorry!.
df = df.sort_values(['key', 'date'])

# Each data set has a unique name. Let's create groups.
g = df.groupby('key')

# The parameter 'day' is the number of days since the first 
# day of data collection for the group.
df['day'] = g['date'].transform(lambda x: (pd.to_datetime(
    x) - min(pd.to_datetime(x)))).apply(lambda x: x.days)


# The Corona Data Scraper Data contains cumulative totals
# for each group. We want _new_ amounts per the previous
# (reporting) day. So we 'shift' or lag the data by
# one.
df["positives_lagged"] = g.positives.shift(1)
df["death_lagged"] = g.death.shift(1)
df['active_lagged'] = g.active.shift(1)
df['tests_lagged'] = g.tests.shift(1)

# And subtract the current number from the previous reported
# day, filling in the first number with data from the first
# row of the group. Other N/As will remain N/A.
df['new_positives'] = df.positives.sub(
    df.positives_lagged).fillna(df.positives)
df['new_active'] = df.active.sub(df.active_lagged).fillna(df.active)
df['new_death'] = df.death.sub(df.death_lagged).fillna(df.death)
df['new_tests'] = df.tests.sub(df.tests_lagged).fillna(df.tests)

# We'll drop the unneeded lag data.
df.drop(['positives_lagged', 'death_lagged', 'active_lagged', 'tests_lagged'], inplace=True, axis=1)
 

# And save it to the data directory.
here = sc.thisdir(__file__)
data_home = os.path.join(here, subfolder)
filepath = sc.makefilepath(filename=outputfile, folder=data_home)
log.info(f"Saving to {filepath}")
df.to_csv(filepath)
log.info(f"Script complete")
