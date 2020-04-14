'''
This script creates a single file containing all the scraped 
data from the Covid Data Project Data Scraper.
https://covidtracking.com
'''

from data.scraper import Scraper
import datetime as dt

def covid_tracking_date_to_date(d):
    return dt.date((d // 10000), ((d % 1000) // 100), (d % 1000) % 100)

class CovidTrackingProjectScraper(Scraper):
    def create_date(self):
        self.df['date'] = self.df.date.apply(covid_tracking_date_to_date)


class CovidUSTrackingProjectScraper(CovidTrackingProjectScraper):
    def create_key(self):
        self.df['key'] = 'US'



## Set up parameters 
p = dict()
p['title'] = "Covid Tracking Project Scraper for US states"
p['load_path'] = "https://covidtracking.com/api/v1/states/daily.csv"
# p['load_path'] = "/Users/willf/github/covasim/data/epi_data/ctp_input.csv"

p['output_folder'] = "epi_data"
p['output_filename'] = "covid-tracking-project-us-state-data.csv"

p['renames'] = dict()
p['renames']['state'] = "key"
p['renames']['positiveIncrease'] = "new_positives"
p['renames']['negativeIncrease'] = "new_negatives"
p['renames']['totalTestResultsIncrease'] = "new_tests"
p['renames']['hospitalizedIncrease'] = "new_hospitalized"
p['renames']['deathIncrease'] = "new_death"
p['renames']['inIcuCumulative'] = "cum_in_icu"
p['renames']['hospitalizedCumulative'] = "cum_hospitalized"
p['renames']['onVentilatorCumulative'] = "cum_on_ventilator"

p['cumulative_fields'] = dict()
p['cumulative_fields']['cum_in_icu'] = "num_icu"
p['cumulative_fields']['cum_on_ventilator'] = "num_on_ventilator"


p['fields_to_drop'] = [
    "hash",
    "dateChecked",
    "fips",
    "totalTestResults",
    "posNeg",
    "positive",
    "negative",
    "pending",
    "hospitalizedCurrently",
    "inIcuCurrently",
    "onVentilatorCurrently",
    "recovered",
    "hospitalized",
    "total"
]

# Set up US parameters
parameter_us = dict()
parameter_us['title'] = "Covid Tracking Project Scraper for US states"
parameter_us['load_path'] = "https://covidtracking.com/api/v1/us/daily.csv"
# parameter_us['load_path'] = "/Users/willf/github/covasim/data/epi_data/ctp_us_input.csv"

parameter_us['output_folder'] = "epi_data"
parameter_us['output_filename'] = "covid-tracking-project-us-data.csv"

parameter_us['renames'] = dict()
parameter_us['renames']['positiveIncrease'] = "new_positives"
parameter_us['renames']['negativeIncrease'] = "new_negatives"
parameter_us['renames']['totalTestResultsIncrease'] = "new_tests"
parameter_us['renames']['hospitalizedIncrease'] = "new_hospitalized"
parameter_us['renames']['deathIncrease'] = "new_death"
parameter_us['renames']['inIcuCumulative'] = "cum_in_icu"
parameter_us['renames']['hospitalizedCumulative'] = "cum_hospitalized"
parameter_us['renames']['onVentilatorCumulative'] = "cum_on_ventilator"

parameter_us['cumulative_fields'] = dict()
parameter_us['cumulative_fields']['cum_in_icu'] = "num_icu"
parameter_us['cumulative_fields']['cum_on_ventilator'] = "num_on_ventilator"


parameter_us['fields_to_drop'] = [
    "hash",
    "dateChecked",
    "fips",
    "totalTestResults",
    "posNeg",
    "positive",
    "negative",
    "pending",
    "hospitalizedCurrently",
    "inIcuCurrently",
    "onVentilatorCurrently",
    "recovered",
    "hospitalized",
    "states",
    "total"
]

# Scrape states
CovidTrackingProjectScraper(p).scrape()
# Scrape US
CovidUSTrackingProjectScraper(parameter_us).scrape()


