import os
import covasim.datasets.data_loader as cdt
import json
dirname = os.path.dirname(__file__)

def stub_population_data(path= "mock_datasets/population/neher_labs_translated.json"):
    neherlabs_pop_translated_json_path = os.path.join(dirname,path)
    cdt.NeherLabPop.file_path = lambda self: neherlabs_pop_translated_json_path

def stub_neherlabs_external_call():
    neherlabs_pop_json_path = os.path.join(dirname, "mock_datasets/population/neher_labs.json")
    with open(neherlabs_pop_json_path) as data:
        neherlabs_pop_json = json.load(data)

    cdt.NeherLabPop.fetch_data = lambda self: neherlabs_pop_json
