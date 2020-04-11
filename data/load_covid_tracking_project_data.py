'''
This script creates a single file containing all the scraped 
data from the COVID tracking project.
'''

import os
import sys
import logging
import datetime as dt
import pandas as pd
import sciris as sc

subfolder = 'epi_data'
outputfile = 'covid-tracking-project-data.csv'

log = logging.getLogger("Covid Tracking Project Loader")
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# Read in the Corona Data Scraper Data into a dataframe.

log.info("Loading states timeseries data from covidtracking.com")
df_states = pd.read_csv("https://covidtracking.com/api/v1/states/daily.csv")
df_states['name'] = df_states.state
df_states.drop(['state'], inplace=True, axis=1)
log.info("Loading US timeseries data from covidtracking.com")
df_us = pd.read_csv("https://covidtracking.com/api/v1/us/daily.csv")
df_us['name'] = 'US'
df_us.drop(['states'], inplace=True, axis=1)

# Put them together
df = pd.concat([df_states, df_us], ignore_index=True, sort=False)

# Convert integer date to a datetime date
def covid_tracking_date_to_date(d):
    return dt.date((d // 10000), ((d % 1000) // 100), (d % 1000) % 100)

df.date = df.date.apply(covid_tracking_date_to_date)

# Sort by name and date.
df = df.sort_values(['name', 'date'])

# Each data set has a unique name. Let's create groups.
g = df.groupby('name')

# The parameter 'day' is the number of days since the first
# day of data collection for the group.
df['day'] = g['date'].transform(lambda x: (x - min(x))).apply(lambda x: x.days)

# We'll 'rename' some of the columns to be consistent
# with the parameters file.

df['new_positives'] = df.positiveIncrease
df['new_negatives'] = df.negativeIncrease
df['new_tests'] = df.totalTestResultsIncrease
df['new_death'] = df.deathIncrease

df.drop(['positiveIncrease', 'positiveIncrease',
         'totalTestResultsIncrease', 'deathIncrease'], inplace=True, axis=1)

# The COVID Project contains cumulative totals
# for ICU and ventilator data. We get the daily
# adjusted values and drop the no longer needed
# columns

df['icu_lagged'] = g.inIcuCurrently.shift(1)
df['vent_lagged'] = g.onVentilatorCurrently.shift(1)
df['new_icu'] = df.inIcuCurrently.sub(df.icu_lagged).fillna(df.inIcuCurrently)
df['new_vent'] = df.onVentilatorCurrently.sub(
    df.vent_lagged).fillna(df.onVentilatorCurrently)
df.drop(['icu_lagged', 'vent_lagged'], inplace=True, axis=1)

# And save it to the data directory.
here = sc.thisdir(__file__)
data_home = os.path.join(here, subfolder)
filepath = sc.makefilepath(filename=outputfile, folder=data_home)
log.info(f"Saving to {filepath}")
df.to_csv(filepath)
log.info(f"Script complete")
