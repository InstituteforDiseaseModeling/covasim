'''
This script creates a single file containing all the scraped 
data from the Corona Data Scraper.
'''

from data.scraper import Scraper
import toml

parameter_definitions = f"""

title = "Corona Data Scraper Project Scraper"
load_path = "https://coronadatascraper.com/timeseries.csv"
# load_path = "/Users/willf/github/covasim/data/epi_data/cds_input.csv"

output_folder = "epi_data"
output_filename = "corona_data_scraper.csv"

renames.name = "key"
renames.cases = "cum_positives"
renames.deaths = "cum_death"
renames.tested = "cum_tests"
renames.hospitalized = "cum_hospitalized"
renames.discharged = "cum_discharged"
renames.recovered = "cum_recovered"
renames.active = "cum_active"

cumulative_fields.cum_positives = "positives"
cumulative_fields.cum_death = "death"
cumulative_fields.cum_tests = "tests"
cumulative_fields.cum_hospitalized = "hospitalized"
cumulative_fields.cum_discharged = "discharged"
cumulative_fields.cum_recovered = "recovered"
cumulative_fields.cum_active = "active"


fields_to_drop = [
    "growthFactor", 
    "city",
    "county",  
    "state", 
    "country",
    "lat",
    "long",
    "url",
    "tz",
    "level"
    ]

"""


class CoronaDataScraperScraper(Scraper):
    pass


parameters = toml.loads(parameter_definitions)
CoronaDataScraperScraper(parameters).scrape()
