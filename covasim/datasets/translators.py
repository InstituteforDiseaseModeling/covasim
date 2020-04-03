import json
import numpy as np

def neherlab_translator(country, json_string):
    data = json.loads(json_string)[country]

    result = []
    total_pop = sum(data.values())
    print(total_pop)
    for age, age_pop in data.items():
        if age[-1] == '+':
            val = [int(age[:-1]), 130, age_pop/total_pop]
        else:
            ages = age.split('-')
            val = [int(ages[0]), int(ages[1]), age_pop/total_pop]
        result.append(val)
    return np.array(result)


