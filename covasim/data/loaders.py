'''
Load data
'''

#%% Housekeeping
import numpy as np
import pandas as pd
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
    country_mappings = get_country_mappings()
    mapping = {key.lower(): val.lower() for key, val in mapping.items()}

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


def get_country_mappings():
    country_mappings = {
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

    return country_mappings # Convert to lowercase


def get_country_household_sizes(url=None):
    """
    Load country household size from UN data source.

    Args:
        URL or path to the excel file.

    Returns:
        Dictionary with country name as key and average household size as value.
    """
    url = url or 'https://population.un.org/household/exceldata/population_division_UN_Houseshold_Size_and_Composition_2019.xlsx'
    df_raw: pd.DataFrame = pd.read_excel(url, sheet_name='UN HH Size and Composition 2019', skiprows=4)
    assert len(df_raw) > 1, "Loaded UN Household size."

    # Select and rename columns
    target_columns = ['Country or area', 'Reference date (dd/mm/yyyy)', 'Average household size (number of members)']
    df: pd.DataFrame = df_raw[target_columns].copy()
    df.columns = ['country', 'date', 'size']

    # Convert date column to datetime type and replace nodata with NA.
    df['date'] = df['date'].apply(lambda d: pd.to_datetime(d, format='%d/%m/%Y'))
    df['size'] = df['size'].apply(lambda s: np.nan if isinstance(s, str) and s == '..' else s)
    df = df.dropna()

    # Take the most recent household size for each country.
    df = df.sort_values(by=['country', 'date']).groupby(by=['country']).last()[['size']]
    un_country_households_dict = df.to_dict()['size'].copy()

    # Add to the dictionary commonly used aliases like USA for United States of America.
    us_countries = [k.lower() for k in un_country_households_dict]
    country_mappings_dict = get_country_mappings()
    for alias, name in country_mappings_dict.items():
        if name.lower() in us_countries:
            un_country_households_dict[alias] = un_country_households_dict[name]

    return un_country_households_dict
