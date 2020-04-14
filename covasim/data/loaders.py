'''
Load data
'''

#%% Housekeeping
import numpy as np
import sciris as sc
import pandas as p
from . import country_age_distributions as cad

__all__ = ['get_age_distribution', 'get_us_state_age_distribution']


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


def get_us_state_age_distribution(state):
    '''
    Load age distribution for a given US state.

    Args:
        location (str or array): name of the state to load the age distribution for

    Returns:
        age_data (dataframe): Pandas data frame of age distributions
    '''

    data = p.read_pickle('data/us_census.pickle')
    data['State'] = data['State'].str.lower()

    states = data.State.values

    if state is None:
        state = states
    else:
        state = sc.promotetolist(state)

    age_groups = ["0-5", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-29", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85+"]
    result = {}
    for loc in state:
        state_data = data.loc[data['State'] == loc.lower()]
        local_pop = []
        for age in age_groups:
            percent = state_data[age].values[0]
            if age[-1] == '+':
                val = [int(age[:-1]), 99, percent]
            else:
                ages = age.split('-')
                val = [int(ages[0]), int(ages[1]), percent]
            local_pop.append(val)
        result[loc] = np.array(local_pop)

    if len(state) == 1:
        result = result[loc]

    return result
