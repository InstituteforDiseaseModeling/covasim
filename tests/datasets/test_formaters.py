'''
Test the people class
'''

#%% Imports and settings
import sciris as sc
import covasim.datasets.translators as cdt
import numpy as np
import json
from unittest.mock import MagicMock
import os

dirname = os.path.dirname(__file__)

neherlabs_pop_json_path = os.path.join(dirname, "mock_datasets/population/neher_labs.json")
with open(neherlabs_pop_json_path) as data:
    neherlabs_pop_json = json.load(data)

def test_transform_neherlab_data():
    sc.heading('Test the transformation for the neherlab covid19 scenarios dataset')

    json_string = """{
      "Afghanistan": {
        "0-9": 100,
        "10-19": 200,
        "20-29": 300,
        "30-39": 450,
        "40-49": 500,
        "50-59": 100,
        "60-69": 200,
        "70-79": 30,
        "80+": 10
      },
      "Albania": {
        "0-9": 333832,
        "10-19": 361777,
        "20-29": 472678,
        "30-39": 390771,
        "40-49": 323020,
        "50-59": 386091,
        "60-69": 330244,
        "70-79": 194668,
        "80+": 84716
      }
    }"""

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
    output = translator.translate(json.loads(json_string))
    np.testing.assert_array_equal(output, expected)

def test_get_country_data():
    translator = cdt.NeherLabPop()
    translator.file_path = MagicMock(return_value=neherlabs_pop_json_path)
    output = translator.data_for_country("Albania")

    expected = neherlabs_pop_json['Albania']
    assert output == expected
