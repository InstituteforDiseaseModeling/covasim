'''
This script creates a single file containing all the scraped 
data from the Corona Data Scraper.
'''

from data.scraper import Scraper

class CoronaDataScraperScraper(Scraper):
    pass

# Set up parameters

parameters = dict()
parameters['title'] = 'Corona Data Scraper Project Scraper'
parameters['load_path'] = 'https://coronadatascraper.com/timeseries.csv'
# parameters['load_path'] = '/Users/willf/github/covasim/data/epi_data/input/cds-input.csv'

parameters['output_folder'] = 'epi_data/corona-data-scraper-project'

parameters['renames'] = dict()
parameters['renames']['name'] = 'key'
parameters['renames']['cases'] = 'cum_positives'
parameters['renames']['deaths'] = 'cum_death'
parameters['renames']['tested'] = 'cum_tests'
parameters['renames']['hospitalized'] = 'cum_hospitalized'
parameters['renames']['discharged'] = 'cum_discharged'
parameters['renames']['recovered'] = 'cum_recovered'
parameters['renames']['active'] = 'cum_active'

parameters['cumulative_fields'] = dict()
parameters['cumulative_fields']['cum_positives'] = 'positives'
parameters['cumulative_fields']['cum_death'] = 'death'
parameters['cumulative_fields']['cum_tests'] = 'tests'
parameters['cumulative_fields']['cum_hospitalized'] = 'hospitalized'
parameters['cumulative_fields']['cum_discharged'] = 'discharged'
parameters['cumulative_fields']['cum_recovered'] = 'recovered'
parameters['cumulative_fields']['cum_active'] = 'active'


parameters['fields_to_drop'] = [
    'growthFactor',
    'city',
    'county',
    'state',
    'country',
    'lat',
    'long',
    'url',
    'tz',
    'level'
]

CoronaDataScraperScraper(parameters).scrape()
