import requests
import os
import json
import numpy as np
from pathlib import Path
dirname = os.path.dirname(__file__)


class NeherLabPop():
    URL="https://raw.githubusercontent.com/neherlab/covid19_scenarios/master/src/assets/data/country_age_distribution.json"
    FILENAME="population"

    def __init__(self):
        # Make the file folder if it doesn't exist
        Path(os.path.dirname(self.file_path())).mkdir(parents=True, exist_ok=True)

    ##
    # countries: This data source defines its data at the country level
    # this will return the countries in the source file.
    #
    def countries(self):
        self.load_data()
        return self.data.keys()

    ##
    # data_for_country: returns the demographic
    # age data for the country as a numpy array
    #
    def data_for_country(self, country):
        self.load_data()
        return np.array(self.data[country])

    ##
    # update_data: pull the external data local calls
    # translate with the data from the external source
    # and saves it locally. Only needs to be called
    # once ever, unless the external source changes
    #
    def update_data(self):
        data_json = self.fetch_data()
        self.delete_data()
        translated = self.translate(data_json)
        with open(self.file_path(), "a") as outfile:
            json.dump(translated, outfile)

    ##
    # fetch_data: gets eternal data and casts
    # it to json. May need pandas or somethig else
    # eventually.
    #
    def fetch_data(self):
        r = requests.get(self.URL)
        return r.json()

    ##
    # delete_data: delete the locally stored trnaslated file.
    #
    def delete_data(self):
        if os.path.exists(self.file_path()):
            os.remove(self.file_path())

    ##
    # file_path: path to the locally stored translated data
    #
    def file_path(self):
        return os.path.join(dirname,"../../data/{0}.json".format(self.FILENAME))

    ##
    # Load the locally stored translated data into cache.
    #
    def load_data(self):
        if hasattr(self, 'data'):
            return
        with open(self.file_path()) as datafile:
            strdata = json.load(datafile)
        self.data = strdata

    ##
    # translate: takes the neherlab formate and turns it
    # into our prefered local format
    #
    def translate(self, json=False):
        result = {}
        for location in json:
            country = location["country"]
            age_distribution = location["ageDistribution"]
            total_pop = sum(age_distribution.values())
            local_pop = []

            for age, age_pop in age_distribution.items():
                if age[-1] == '+':
                    val = [int(age[:-1]), 130, age_pop/total_pop]
                else:
                    ages = age.split('-')
                    val = [int(ages[0]), int(ages[1]), age_pop/total_pop]
                local_pop.append(val)
            result[country] = local_pop

        return result


##
# Looks up where the country population data is stored, using a reverse index
# and loads the data for the specified country.
#
def load_country_pop(country):
    return NeherLabPop().data_for_country(country)

def available_countries():
    # Hmm, I think we will need to support some default settings for the web
    # UI https://github.com/InstituteforDiseaseModeling/covasim/blob/adc95ec589cbf0b06cbca8d8ba2b2a536a549ee7/covasim/webapp/cova_app.py#L38-L69
    # May need to think about how to create this...
    return NeherLabPop().countries()
