'''
Load data
'''

#%% Housekeeping
import numpy as np
import sciris as sc
from . import country_age_distributions as cad

__all__ = ['get_age_distribution']


def get_age_distribution(location=None):
    '''
    Load age distribution for a given country or countries.

    Args:
        location (str or list): name of the country or countries to load the age distribution for

    Returns:
        age_data (array): Numpy array of age distributions
    '''

    # Load the raw data
    json = cad.get_country_age_distributions()
    countries = [entry["country"].lower() for entry in json] # Pull out available countries

    # Set parameters
    max_age = 99
    if location is None:
        location = countries
    else:
        location = sc.promotetolist(location)

    # Define a mapping for common mistakes
    mapping = {
       'Bolivia':        'Bolivia (Plurinational State of)',
       'Burkina':        'Burkina Faso',
       'Cape Verde':     'Cabo Verdeo',
       'Hong Kong':      'China, Hong Kong Special Administrative Region',
       'Macao':          'China, Macao Special Administrative Region',
       "Cote d'Ivore":   'Côte d’Ivoire',
       'DRC':            'Democratic Republic of the Congo',
       'Iran':           'Iran (Islamic Republic of)',
       'Laos':           "Lao People's Democratic Republic",
       'Micronesia':     'Micronesia (Federated States of)',
       'Korea':          'Republic of Korea',
       'South Korea':    'Republic of Korea',
       'Moldova':        'Republic of Moldova',
       'Russia':         'Russian Federation',
       'Palestine':      'State of Palestine',
       'Syria':          'Syrian Arab Republic',
       'Taiwan':         'Taiwan Province of China',
       'Macedonia':      'The former Yugoslav Republic of Macedonia',
       'UK':             'United Kingdom of Great Britain and Northern Ireland',
       'United Kingdom': 'United Kingdom of Great Britain and Northern Ireland',
       'Tanzania':       'United Republic of Tanzania',
       'USA':            'United States of America',
       'United States':  'United States of America',
       'Venezuela':      'Venezuela (Bolivarian Republic of)',
       'Vietnam':        'Viet Nam',
        }
    mapping = {key.lower():val.lower() for key,val in mapping.items()} # Convert to lowercase

    result = {}
    for loc in location:
        loc = loc.lower()
        if loc in mapping:
            loc = mapping[loc]
        try:
            ind = countries.index(loc.lower())
            entry = json[ind]
        except ValueError:
            suggestions = sc.suggest(loc, countries, n=4)
            errormsg = f'Location "{loc}" not recognized, did you mean {suggestions}?'
            raise ValueError(errormsg)
        age_distribution = entry["ageDistribution"]
        total_pop = sum(age_distribution.values())
        local_pop = []

        for age, age_pop in age_distribution.items():
            if age[-1] == '+':
                val = [int(age[:-1]), max_age, age_pop/total_pop]
            else:
                ages = age.split('-')
                val = [int(ages[0]), int(ages[1]), age_pop/total_pop]
            local_pop.append(val)
        result[loc] = np.array(local_pop)

    if len(location) == 1:
        result = result[loc]

    return result
