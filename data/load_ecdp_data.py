from data.scraper import Scraper
import pandas as pd
import os
import toml

import sciris as sc
subfolder = "epi_data"
here = sc.thisdir(__file__)
data_home = os.path.join(here, subfolder)

parameter_definitions = f"""

title = "European Centre for Disease Prevention and Control Covid-19 Data Scraper"
# load_path = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
load_path = "/Users/willf/github/covasim/data/epi_data/epi_test.csv"

output_folder = "epi_data"
output_filename = "ecdp_data.csv"

renames.countriesAndTerritories = "key"
renames.cases = "new_positives"
renames.deaths = "new_death"
renames.popData2018 = "population"

fields_to_drop = [
    "dateRep", 
    "month",
    "year",  
    "geoId", 
    "countryterritoryCode"
    ]

"""



class ECDPScraper(Scraper):
    def create_date(self):
        self.df["date"] = pd.to_datetime(self.df[["year", "month", "day"]])


parameters = toml.loads(parameter_definitions)
ECDPScraper(parameters).scrape()

