# This file is for testing purposes only - delete afterwards

import covasim as cv
import sciris as sc
import random
import json

if __name__ == "__main__":

    with open('scenario_result.json') as f: 
        data = json.load(f)
    
    #data is a dict, which is unordered. 
    # results - cum_inf - scenario - best

    # get the scenario keys / names
    scenarios = data['scenarios'].keys()
    for key in scenarios:
        cum_infections = list(data['results']['cum_infections'][key]['best'])[-1]
        cum_recoveries = list(data['results']['cum_recoveries'][key]['best'])[-1]
        cum_symptomatic = list(data['results']['cum_symptomatic'][key]['best'])[-1]
        cum_severe = list(data['results']['cum_severe'][key]['best'])[-1]
        cum_critical = list(data['results']['cum_critical'][key]['best'])[-1]
        cum_deaths = list(data['results']['cum_deaths'][key]['best'])[-1]

        f = open("baselineExperiment.csv","a+")
        f.write(str(20) + "," + str(key) + ","
                + str(cum_infections) + "," 
                + str(cum_recoveries) + ","
                + str(cum_symptomatic) + ","
                + str(cum_severe) + ","
                + str(cum_critical) + ","
                + str(cum_deaths) + "\n")

        f.close()