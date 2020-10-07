import synthpops as sp
import covasim as cova
import covasim as cv
import sciris as sc

pars = {
    'pop_size': 40000,
    'pop_type': 'synthpops',
}

sim = cv.Sim(pars)



popdict = cv.make_people(sim,
                         generate=True,
                         save_pop=True,popfile='voriPop.pop',
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
