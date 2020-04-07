##
# Update Sources
#
# This file will update all the population sources by
# Pulling in different sources, translating them to the prefered
# format and storing them locally.
#
# Add new loaders by addng the classes to sources and creating a
# new dataloader that inherits from the Dataloader class.
#
# Run with python covasim/datasets/update_sources.py
#
import requests
import os
import json
from covasim.datasets.data_loader import load_country_pop
from covasim.datasets.data_loader import NeherLabPop
dirname = os.path.dirname(__file__)

def load_sources():
    print("Updating data sources")
    countries = {}
    source = NeherLabPop()
    source.update_data()
    for country in source.countries():
        countries[country] = source.__class__.__name__

if __name__ == '__main__':
    load_sources()
    print(load_country_pop("Albania"))
