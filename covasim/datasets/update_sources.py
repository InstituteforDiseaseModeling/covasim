import requests
import os
import json
import covasim.datasets.translators as t
dirname = os.path.dirname(__file__)
mapper_file_path = os.path.join(dirname,"sources/lookup.json")


def load_country_pop(country):
    with open(mapper_file_path) as f:
        mapper = json.load(f)
    mapper = json.loads(mapper)
    data_loader = eval("t.{0}".format(mapper[country]))()
    return data_loader.data_for_country(country)


sources = [
        t.NeherLabPop(),
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
