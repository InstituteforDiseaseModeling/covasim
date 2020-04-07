import requests
import os
import json
dirname = os.path.dirname(__file__)
mapper_file_path = os.path.join(dirname,"sources/lookup.json")


class DataLoader:
    def __init__(self):
        pass

    def update_data(self):
        data_json = self.fetch_data()
        self.delete_data()
        translated = self.translate(data_json)
        with open(self.file_path(), "a") as outfile:
            json.dump(translated, outfile)

    def fetch_data(self):
        r = requests.get(self.URL)
        return r.json()

    def delete_data(self):
        if os.path.exists(self.file_path()):
            os.remove(self.file_path())

    def file_path(self):
        return os.path.join(dirname,"sources/population/{0}.json".format(self.FILENAME))

    def load_data(self):
        if hasattr(self, 'data'):
            return
        with open(self.file_path()) as datafile:
            strdata = json.load(datafile)
        self.data = strdata

    def translate(self, json):
        raise NotImplementedError

class NeherLabPop(DataLoader):
    URL="https://raw.githubusercontent.com/neherlab/covid19_scenarios/master/src/assets/data/country_age_distribution.json"
    FILENAME="neherlab"

    def countries(self):
        self.load_data()
        return self.data.keys()

    def data_for_country(self, country):
        self.load_data()
        return self.data[country]

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


def load_country_pop(country):
    with open(mapper_file_path) as f:
        mapper = json.load(f)
    mapper = json.loads(mapper)
    data_loader = eval("{0}".format(mapper[country]))()
    return data_loader.data_for_country(country)

