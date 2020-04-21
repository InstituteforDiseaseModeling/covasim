'''
Load data
'''

#%% Housekeeping
import numpy as np
import pandas as pd
import sciris as sc
from . import country_age_data as cad
from . import state_age_data as sad
from . import household_size_data as hsd

__all__ = ['get_country_aliases', 'map_entries', 'get_age_distribution', 'get_household_size']


def get_country_aliases():
    ''' Define aliases for countries with odd names in the data '''
    country_mappings = {
       'Bolivia':        'Bolivia (Plurinational State of)',
       'Burkina':        'Burkina Faso',
       'Cape Verde':     'Cabo Verdeo',
       'Hong Kong':      'China, Hong Kong Special Administrative Region',
       'Macao':          'China, Macao Special Administrative Region',
       "Cote d'Ivore":   'Côte d’Ivoire',
       "Ivory Coast":    'Côte d’Ivoire',
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

    return country_mappings # Convert to lowercase


def map_entries(json, location, which):
    '''
    Find a match between the JSON file and the provided location(s).

    Args:
        json (list or dict): the data being loaded
        location (list or str): the list of locations to pull from
        which (str): either 'age' for age data or 'household' for household size distributions

    '''

    # The data have slightly different formats: list of dicts or just a dict
    if which == 'age':
        countries = [entry["country"].lower() for entry in json] # Pull out available countries
    else:
        countries = [key.lower() for key in json.keys()]

    # Set parameters
    if location is None:
        location = countries
    else:
        location = sc.promotetolist(location)

    # Define a mapping for common mistakes
    mapping = get_country_aliases()
    mapping = {key.lower(): val.lower() for key, val in mapping.items()}

    entries = {}
    for loc in location:
        lloc = loc.lower()
        if lloc not in countries and lloc in mapping:
            lloc = mapping[lloc]
        try:
            ind = countries.index(lloc)
            if which == 'age':
                entry = json[ind]
            else:
                entry = list(json.values())[ind]
            entries[loc] = entry
        except ValueError as E:
            suggestions = sc.suggest(loc, countries, n=4)
            if suggestions:
                errormsg = f'Location "{loc}" not recognized, did you mean {suggestions}? ({str(E)})'
            else:
                errormsg = f'Location "{loc}" not recognized ({str(E)})'
            raise ValueError(errormsg)

    return entries


def get_age_distribution(location=None):
    '''
    Load age distribution for a given country or countries.

    Args:
        location (str or list): name of the country or countries to load the age distribution for

    Returns:
        age_data (array): Numpy array of age distributions, or dict if multiple locations
    '''

    # Load the raw data
    json = add.age_distribution_data()
    entries = map_entries(json, location, which='age')

    max_age = 99
    result = {}
    for loc,entry in entries.items():
        age_distribution = entry["ageDistribution"]
        total_pop = sum(list(age_distribution.values()))
        local_pop = []

        for age, age_pop in age_distribution.items():
            if age[-1] == '+':
                val = [int(age[:-1]), max_age, age_pop/total_pop]
            else:
                ages = age.split('-')
                val = [int(ages[0]), int(ages[1]), age_pop/total_pop]
            local_pop.append(val)
        result[loc] = np.array(local_pop)

    if len(result) == 1:
        result = list(result.values())[0]

    return result



def get_us_state_age_distribution(state):
    '''
    Load age distribution for a given US state.

    Args:
        location (str or array): name of the state to load the age distribution for

    Returns:
        age_data (dataframe): Pandas data frame of age distributions
    '''

    data = sad.state_age_distributions()
    states = data.keys()

    if state is None:
        state = states
    else:
        state = sc.promotetolist(state)

    result = {}
    for loc in state:
        try:
            state_data = data[loc.lower()]
        except KeyError:
            suggestions = sc.suggest(loc.lower(), states, n=4)
            errormsg = f'Location "{loc}" not recognized, did you mean {suggestions}?'
            raise KeyError(errormsg)

        local_pop = []
        for age in state_data:
            percent = state_data[age]
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


def get_household_size(location=None):
    '''
    Load household size distribution for a given country or countries.

    Args:
        location (str or list): name of the country or countries to load the age distribution for

    Returns:
        house_size (float): Size of household, or dict if multiple locations
    '''
    # Load the raw data
    json = hsd.household_size_data()

    result = map_entries(json, location, which='household')
    if len(result) == 1:
        result = list(result.values())[0]

    return result
