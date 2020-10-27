import covasim as cv
import sciris as sc
import random
import json


pars = {
    'pop_size': 40000,
    'pop_type': 'synthpops',
    #'rand_seed': random.randint(0, 99999),
}

sim = cv.Sim(pars)

# it always creates the same population
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
    n_days       = 30,       # Number of days to simulate
    pop_scale = 10,
    n_beds_icu = 30,
    n_beds_hosp = 700,
    contacts = 0.5
)

if __name__ == "__main__":
    mysim = cv.sim.Sim(pars=pars_dict, load_pop=True, popfile='baseline_exp.pop')
    #multiSim = cv.MultiSim(sims=mysim,n_runs=1)
    #multiSim.run()
    
    mysim.run()
    mysim.to_json('baseline_exp_results.json')

    with open('baseline_exp_results.json') as f: 
        data = json.load(f)
    
    print(data['summary']['cum_infectious'])

    