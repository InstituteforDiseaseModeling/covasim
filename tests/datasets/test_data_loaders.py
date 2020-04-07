'''
Test the people class
'''

#%% Imports and settings
import sciris as sc
import covasim.datasets.data_loader as cdt
import numpy as np
import json
import os
from dataset_helper import *

dirname = os.path.dirname(__file__)

neherlabs_pop_json_path = os.path.join(dirname, "mock_datasets/population/neher_labs.json")
neherlabs_pop_translated_json_path = os.path.join(dirname, "mock_datasets/population/neher_labs_translated.json")
test_json_path = os.path.join(dirname, "mock_datasets/population/test.json")
with open(neherlabs_pop_translated_json_path) as data:
    neherlabs_pop_translated_json = json.load(data)

neher_labs_raw_pop = open(neherlabs_pop_json_path, "r").read()

def test_transform_neherlab_data():
    sc.heading('Test the transformation for the neherlab covid19 scenarios dataset')
    translator = cdt.NeherLabPop()
    output = translator.translate(json.loads(neher_labs_raw_pop))
    np.testing.assert_array_equal(output, neherlabs_pop_translated_json)

def test_get_country_data():
    stub_population_data()
    translator = cdt.NeherLabPop()
    output = translator.data_for_country("Albania")

    expected = np.array(neherlabs_pop_translated_json['Albania'])
    np.testing.assert_array_equal(output, expected)

def test_load_country_data():
    stub_population_data(path=test_json_path)
    stub_neherlabs_external_call()
    translator = cdt.NeherLabPop()
    translator.update_data()
    output = translator.data_for_country("Albania")
    expected = np.array(neherlabs_pop_translated_json['Albania'])
    np.testing.assert_array_equal(output, expected)
    # Set back to the normal stub so we don't break future tests
    stub_population_data()
    os.remove(test_json_path)



