import requests
import os
import json
from covasim.datasets.data_loader import load_country_pop
from covasim.datasets.data_loader import NeherLabPop
dirname = os.path.dirname(__file__)
mapper_file_path = os.path.join(dirname,"sources/lookup.json")




sources = [
    NeherLabPop(),
]

def load_sources():
    countries = {}

    for source in sources:
        source.update_data()
        for country in source.countries():
            countries[country] = source.__class__.__name__

    os.remove(mapper_file_path)
    with open(mapper_file_path, "a") as outfile:
            json.dump(json.dumps(countries), outfile)

if __name__ == '__main__':
    load_sources()
    print(load_country_pop("Albania"))
