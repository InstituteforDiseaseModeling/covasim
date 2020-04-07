'''
Test the people class
'''

#%% Imports and settings
import sciris as sc
import covasim.datasets.data_loader as cdt
import numpy as np
import json
from unittest.mock import MagicMock
import os

dirname = os.path.dirname(__file__)

neherlabs_pop_json_path = os.path.join(dirname, "mock_datasets/population/neher_labs.json")
neherlabs_pop_translated_json_path = os.path.join(dirname, "mock_datasets/population/neher_labs_translated.json")
test_json_path = os.path.join(dirname, "mock_datasets/population/test.json")
with open(neherlabs_pop_json_path) as data:
    neherlabs_pop_json = json.load(data)
with open(neherlabs_pop_translated_json_path) as data:
    neherlabs_pop_translated_json = json.load(data)

neher_labs_raw_pop = open(neherlabs_pop_json_path, "r").read()

def test_transform_neherlab_data():
    sc.heading('Test the transformation for the neherlab covid19 scenarios dataset')


    expected = { "Afghanistan": [
        [0, 9, 0.05291005291005291],
        [10, 19, 0.10582010582010581],
        [20, 29, 0.15873015873015872],
        [30, 39, 0.23809523809523808],
        [40, 49, 0.26455026455026454],
        [50, 59, 0.05291005291005291],
        [60, 69, 0.10582010582010581],
        [70, 79, 0.015873015873015872],
        [80, 130, 0.005291005291005291],
    ],
    "Albania": [
         [0, 9, 0.11600262283962351],
         [10, 19, 0.125713175738247],
         [20, 29, 0.16424994535750784],
         [30, 39, 0.13578824357659697],
         [40, 49, 0.11224558229784797],
         [50, 59, 0.13416199961289835],
         [60, 69, 0.1147558358007879],
         [70, 79, 0.06764479912933401],
         [80, 130, 0.029437795647156487]
    ]
    }

    translator = cdt.NeherLabPop()
    output = translator.translate(json.loads(neher_labs_raw_pop))
    np.testing.assert_array_equal(output, expected)

def test_get_country_data():
    translator = cdt.NeherLabPop()
    translator.file_path = MagicMock(return_value=neherlabs_pop_translated_json_path)
    output = translator.data_for_country("Albania")

    expected = np.array(neherlabs_pop_translated_json['Albania'])
    np.testing.assert_array_equal(output, expected)

def test_load_country_data():
    translator = cdt.NeherLabPop()
    json_response = json.loads(neher_labs_raw_pop)
    translator.fetch_data = MagicMock(return_value=json_response)
    translator.file_path = MagicMock(return_value=test_json_path)
    translator.update_data()
    output = translator.data_for_country("Albania")
    expected = np.array(neherlabs_pop_translated_json['Albania'])
    np.testing.assert_array_equal(output, expected)
    os.remove(test_json_path)



