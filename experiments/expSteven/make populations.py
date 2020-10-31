import covasim as cv
import sciris as sc
from experiments.SimVorarlberg.pars import pars
import random

if __name__ == '__main__':

    sim = cv.Sim(pars, rand_seed=1)

    people = cv.make_synthpop(sim=sim, max_contacts=None, generate=True, with_industry_code=False, with_facilities=False,
                    use_two_group_reduction=True, average_LTCF_degree=20, ltcf_staff_age_min=20, ltcf_staff_age_max=60,
                    with_school_types=False, school_mixing_type='random', average_class_size=20, inter_grade_mixing=0.1,
                    average_student_teacher_ratio=20, average_teacher_teacher_degree=3, teacher_age_min=25, teacher_age_max=75,
                    with_non_teaching_staff=False,
                    average_student_all_staff_ratio=15, average_additional_staff_degree=20, staff_age_min=20, staff_age_max=75,
                    country_location='austria', state_location='Vorarlberg', location='Vorarlberg', sheet_name='Austria')



    filepath = sc.makefilepath(filename='pop1.pop')
    cv.save(filepath, people)

    sim = cv.Sim(pars, rand_seed=2)

    people = cv.make_synthpop(sim=sim, max_contacts=None, generate=True, with_industry_code=False, with_facilities=False,
                    use_two_group_reduction=True, average_LTCF_degree=20, ltcf_staff_age_min=20, ltcf_staff_age_max=60,
                    with_school_types=False, school_mixing_type='random', average_class_size=20, inter_grade_mixing=0.1,
                    average_student_teacher_ratio=20, average_teacher_teacher_degree=3, teacher_age_min=25, teacher_age_max=75,
                    with_non_teaching_staff=False,
                    average_student_all_staff_ratio=15, average_additional_staff_degree=20, staff_age_min=20, staff_age_max=75,
                    country_location='austria', state_location='Vorarlberg', location='Vorarlberg', sheet_name='Austria')



    filepath = sc.makefilepath(filename='pop2.pop')
    cv.save(filepath, people)

    sim = cv.Sim(pars, rand_seed=3)

    people = cv.make_synthpop(sim=sim, max_contacts=None, generate=True, with_industry_code=False, with_facilities=False,
                    use_two_group_reduction=True, average_LTCF_degree=20, ltcf_staff_age_min=20, ltcf_staff_age_max=60,
                    with_school_types=False, school_mixing_type='random', average_class_size=20, inter_grade_mixing=0.1,
                    average_student_teacher_ratio=20, average_teacher_teacher_degree=3, teacher_age_min=25, teacher_age_max=75,
                    with_non_teaching_staff=False,
                    average_student_all_staff_ratio=15, average_additional_staff_degree=20, staff_age_min=20, staff_age_max=75,
                    country_location='austria', state_location='Vorarlberg', location='Vorarlberg', sheet_name='Austria')



    filepath = sc.makefilepath(filename='pop3.pop')
    cv.save(filepath, people)