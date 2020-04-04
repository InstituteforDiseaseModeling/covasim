import requests
import os
import json
import covasim.datasets.translators as t
dirname = os.path.dirname(__file__)
mapper_file_path = os.path.join(dirname,"sources/lookup.json")

class DataLoader:
    def __init__(self):
        self.load_data()

    def update_data(self):
        r = requests.get(self.URL)
        data_json = r.json()
        os.remove(self.file_path())
        with open(self.file_path(), "a") as outfile:
            json.dump(data_json, outfile)
        self.data = data_json

    def file_path(self):
        return os.path.join(dirname,"sources/population/{0}.json".format(self.FILENAME))

    def load_data(self, path=False):
        with open(self.file_path()) as datafile:
            strdata = json.load(datafile)
        self.data = strdata

    def data_for_country(self, country):
        func = self.translator()
        return func(country, self.data)



class NeherLabPop(DataLoader):
    URL="https://raw.githubusercontent.com/neherlab/covid19_scenarios/master/src/assets/data/country_age_distribution.json"
    FILENAME="neherlab"

    def countries(self):
        return self.data.keys()

    def translator(self, translator=t.neherlab_translator):
        return translator



def load_country_pop(country):
    with open(mapper_file_path) as f:
        mapper = json.load(f)
    mapper = json.loads(mapper)
    loader = eval(mapper[country])()
    return loader.data_for_country(country)


sources = [
        NeherLabPop(),
]

countries = {}

for source in sources:
    source.update_data()
    for country in source.countries():
        countries[country] = source.__class__.__name__

os.remove(mapper_file_path)
with open(mapper_file_path, "a") as outfile:
        json.dump(json.dumps(countries), outfile)

print(load_country_pop("Albania"))
