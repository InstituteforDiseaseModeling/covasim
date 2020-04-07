import os
import covasim.datasets.data_loader as cdt
import json
dirname = os.path.dirname(__file__)

##
# The external data is not consistent and changes over time, so we cannot safely test
# against it. These are helper methods to mock out api or data fetches for external systems to
# test data translation and also stubs to swith to using consistent locally cached data.

##
# Stub out the locally cached population data so it is consistent and
# present when not in a populated env such as a test.
#
def stub_population_data(path= "mock_datasets/population/neher_labs_translated.json"):
    neherlabs_pop_translated_json_path = os.path.join(dirname,path)
    cdt.NeherLabPop.file_path = lambda self: neherlabs_pop_translated_json_path

##
# Stub out the call to get the Neherlabs population data
# instead use a local version of the data that can replicate
# their data format
#
def stub_neherlabs_external_call():
    neherlabs_pop_json_path = os.path.join(dirname, "mock_datasets/population/neher_labs.json")
    with open(neherlabs_pop_json_path) as data:
        neherlabs_pop_json = json.load(data)

    cdt.NeherLabPop.fetch_data = lambda self: neherlabs_pop_json
