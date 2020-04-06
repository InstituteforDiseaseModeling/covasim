import json
import numpy as np
from covasim.datasets.data_loader import DataLoader

class NeherLabPop(DataLoader):
    URL="https://raw.githubusercontent.com/neherlab/covid19_scenarios/master/src/assets/data/country_age_distribution.json"
    FILENAME="neherlab"

    def countries(self):
        return self.data.keys()

    def _for_country(self, country):
        return self.data[country]

    def translate(self, json=False):
        result = {}
        for key in json:
            data = json[key]
            total_pop = sum(data.values())
            local_pop = []
            for age, age_pop in data.items():
                if age[-1] == '+':
                    val = [int(age[:-1]), 130, age_pop/total_pop]
                else:
                    ages = age.split('-')
                    val = [int(ages[0]), int(ages[1]), age_pop/total_pop]
                local_pop.append(val)
            result[key] = local_pop

        return result


