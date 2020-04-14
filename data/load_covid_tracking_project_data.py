'''
This script creates a single file containing all the scraped 
data from the Corona Data Scraper.
'''

from data.scraper import Scraper
import toml
import datetime as dt

parameter_definitions = f"""

title = "Covid Tracking Project Scraper"
# load_path = "https://coronadatascraper.com/timeseries.csv"
load_path = "/Users/willf/github/covasim/data/epi_data/ctp_input.csv"

output_folder = "epi_data"
output_filename = "covid-tracking-project-data.csv"

renames.state = "key"
renames.positiveIncrease = "new_positives"
renames.negativeIncrease = "new_negatives"
renames.totalTestResultsIncrease = "new_tests"
renames.hospitalizedIncrease = "new_hospitalized"
renames.deathIncrease = "new_death"
renames.inIcuCumulative = "cum_in_icu"
renames.hospitalizedCumulative = "cum_hospitalized"
renames.onVentilatorCurrently = "cum_on_ventilator"

cumulative_fields.cum_in_icu = "num_icu"
cumulative_fields.cum_on_ventilator = "num_on_ventilator"


fields_to_drop = [
    "hash", 
    "dateChecked",
    "fips",  
    "totalTestResults",
    "posNeg"
    ]

"""


def covid_tracking_date_to_date(d):
    return dt.date((d // 10000), ((d % 1000) // 100), (d % 1000) % 100)

class CovidTrackingProjectScraper(Scraper):
    def create_date(self):
        self.df['date'] = self.df.date.apply(covid_tracking_date_to_date)


parameters = toml.loads(parameter_definitions)
CovidTrackingProjectScraper(parameters).scrape()
