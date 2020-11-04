import covasim as cv
import sciris as sc
import random
import json

basepars = dict(
  pop_size = 40000,
  verbose = False,
)

# Scenario metaparameters
metapars = dict(
    n_runs    = 1, # Number of parallel runs; change to 3 for quick, 11 for real
    noise     = 0.1, # Use noise, optionally
    noisepar  = 'beta',
    rand_seed = None,
    quantiles = {'low':0.1, 'high':0.9},
)

def makeRunsAndSaveResultsAsCSV(fileName = "baselineExperiment.csv", numberOfRandomPop=2, scenarios: dict = None):

    f = open(fileName,"w+")
    f.write("pop_seed,scenario,cum_infections,cum_infectious,cum_recoveries,cum_symptomatic,cum_severe,cum_critical,cum_deaths\n")
    f.close

    for i in range(numberOfRandomPop):
        pop_seed = random.randint(0, 999999)
        pop_seed = i # delete afterwards

        pars = {
            'pop_size': 40000,
            'pop_type': 'synthpops',
            'rand_seed': i #was pop_seed
        }

        sim = cv.Sim(pars)

        popdict = cv.make_people(sim,
                                generate=True,
                                save_pop=True,popfile='baseline_exp.pop',
                                location='Vorarlberg',
                                state_location='Vorarlberg',
                                country_location='Austria',
                                sheet_name='Austria',
                                with_school_types=False,
                                school_mixing_type='random',
                                average_class_size=20,
                                inter_grade_mixing=0.1,
                                average_student_teacher_ratio=16,
                                average_teacher_teacher_degree=3,
                                teacher_age_min=25,
                                teacher_age_max=70,
                                average_student_all_staff_ratio=15,
                                average_additional_staff_degree=20,
                                staff_age_min=20,
                                staff_age_max=70,
                                verbose=False, 
                                )


        pars_dict = sc.objdict(
            pop_size     = 40e3,    # Population size
            location = "Vorarlberg",
            pop_infected = 10,       # Number of initial infections
            n_days       = 100,       # Number of days to simulate
            pop_scale = 10,
            n_beds_icu = 30,
            n_beds_hosp = 700,
            contacts = 0.5,
        )

        if __name__ == "__main__":
            if scenarios is not None:
                scens = cv.Scenarios(basepars=basepars, metapars=metapars, scenarios=scenarios)
                scens.run(verbose=None)
                scens.to_json('scenario_result.json')

                with open('scenario_result.json') as f: 
                    data = json.load(f)
    
                #data is a dict, which is unordered. 
                # results - cum_inf - scenario - best

                # get the scenario keys / names
                scenarioData = data['scenarios'].keys()
                for key in scenarioData:
                    cum_infections = list(data['results']['cum_infections'][key]['best'])[-1]
                    cum_recoveries = list(data['results']['cum_recoveries'][key]['best'])[-1]
                    cum_symptomatic = list(data['results']['cum_symptomatic'][key]['best'])[-1]
                    cum_severe = list(data['results']['cum_severe'][key]['best'])[-1]
                    cum_critical = list(data['results']['cum_critical'][key]['best'])[-1]
                    cum_deaths = list(data['results']['cum_deaths'][key]['best'])[-1]

                    f = open(fileName,"a+")
                    f.write(str(pop_seed) + "," + str(key) + ","
                            + str(cum_infections) + "," 
                            + str(cum_recoveries) + ","
                            + str(cum_symptomatic) + ","
                            + str(cum_severe) + ","
                            + str(cum_critical) + ","
                            + str(cum_deaths) + "\n")
                    f.close()
                
            else:
                mysim = cv.sim.Sim(pars=pars_dict, load_pop=True, popfile='baseline_exp.pop')
                mysim.run()
                mysim.to_json('baseline_exp_results.json')

                with open('baseline_exp_results.json') as f: 
                    data = json.load(f)
                
                f = open(fileName,"a+")
                f.write(str(pop_seed) + "," + "no_scenario" + ","
                    + str(data['summary']['cum_infections']) + "," 
                    + str(data['summary']['cum_recoveries']) + ","
                    + str(data['summary']['cum_symptomatic']) + ","
                    + str(data['summary']['cum_severe']) + ","
                    + str(data['summary']['cum_critical']) + ","
                    + str(data['summary']['cum_deaths']) + "\n")
                f.close()

# scenarios
start_day = '2020-04-04'
scenarios = {'baseline': {
              'name':'Baseline',
              'pars': {
                  'interventions': None,
                  }
              },
            'distance': {
              'name':'Social distancing',
              'pars': {
                  'interventions': cv.change_beta(days=start_day, changes=0.7)
                  }
              },
            'ttq': {
              'name':'Test-trace-quarantine',
              'pars': {
                  'interventions': [
                        cv.test_prob(start_day=start_day, symp_prob=0.2, asymp_prob=0.05, test_delay=1.0),
                        cv.contact_tracing(start_day=start_day, trace_probs=0.8, trace_time=1.0),
                    ]
                  }
              },
             }

makeRunsAndSaveResultsAsCSV(fileName = "5pop_baseline0.csv", scenarios=None, numberOfRandomPop=5)

    