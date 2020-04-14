from data.scraper import Scraper
import pandas as pd
import os


class ECDPScraper(Scraper):
    def create_date(self):
        self.df["date"] = pd.to_datetime(self.df[["year", "month", "day"]])


parameters = dict()
parameters['title'] = 'European Centre for Disease Prevention and Control Covid-19 Data Scraper'
parameters['load_path'] = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
# parameters['load_path'] =  '/Users/willf/github/covasim/data/epi_data/input/ecdp-input.csv'

parameters['output_folder'] = 'epi_data'
parameters['output_folder'] = 'epi_data/european-centre-for-disease-prevention-and-control'

parameters['renames'] = dict()
parameters['renames']['countriesAndTerritories'] = 'key'
parameters['renames']['cases'] = 'new_positives'
parameters['renames']['deaths'] = 'new_death'
parameters['renames']['popData2018'] = 'population'

parameters['fields_to_drop'] = [
    'dateRep',
    'month',
    'year',
    'geoId',
    'countryterritoryCode'
]

ECDPScraper(parameters).scrape()

