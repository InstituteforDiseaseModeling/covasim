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
parameters_state = dict()
parameters_state['title'] = "Covid Tracking Project Scraper for US states"
parameters_state['load_path'] = "https://covidtracking.com/api/v1/states/daily.csv"
# parameters_state['load_path'] = "/Users/willf/github/covasim/data/epi_data/input/ctp-state-input.csv"

parameters_state['output_folder'] = "epi_data/covid-tracking-project"

parameters_state['renames'] = dict()
parameters_state['renames']['state'] = "key"
parameters_state['renames']['positiveIncrease'] = "new_positives"
parameters_state['renames']['negativeIncrease'] = "new_negatives"
parameters_state['renames']['totalTestResultsIncrease'] = "new_tests"
parameters_state['renames']['hospitalizedIncrease'] = "new_hospitalized"
parameters_state['renames']['deathIncrease'] = "new_death"
parameters_state['renames']['inIcuCumulative'] = "cum_in_icu"
parameters_state['renames']['hospitalizedCumulative'] = "cum_hospitalized"
parameters_state['renames']['onVentilatorCumulative'] = "cum_on_ventilator"

parameters_state['cumulative_fields'] = dict()
parameters_state['cumulative_fields']['cum_in_icu'] = "num_icu"
parameters_state['cumulative_fields']['cum_on_ventilator'] = "num_on_ventilator"


parameters_state['fields_to_drop'] = [
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
parameter_us['load_path'] = "/Users/willf/github/covasim/data/epi_data/input/ctp-us-input.csv"

parameters_state['output_folder'] = "epi_data/covid-tracking-project"

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
CovidTrackingProjectScraper(parameters_state).scrape()
# Scrape US
CovidUSTrackingProjectScraper(parameter_us).scrape()


