In order to create a valid popfile for Covasim with the data of Vorarlberg with the recent version of synthpops (0.5.1),
you need to change the **make_population** method in **synthpops/synthpops/api.py** by

- deleting lines 83-86::

    83 country_location = 'usa'
    84 state_location = 'Washington'
    85 location = 'seattle_metro'
    86 sheet_name = 'United States of America'

- add the deleted variables as function params at line 18::

    18 def make_population(n=None, max_contacts=None, generate=None, with_industry_code=False, with_facilities=False,
                    use_two_group_reduction=True, average_LTCF_degree=20, ltcf_staff_age_min=20, ltcf_staff_age_max=60,
                    with_school_types=False, school_mixing_type='random', average_class_size=20, inter_grade_mixing=0.1,
                    average_student_teacher_ratio=20, average_teacher_teacher_degree=3, teacher_age_min=25, teacher_age_max=75,
                    with_non_teaching_staff=False,
                    average_student_all_staff_ratio=15, average_additional_staff_degree=20, staff_age_min=20, staff_age_max=75,
                    rand_seed=None, country_location='usa', state_location='Washington', location='seattle_metro', sheet_name='United States of America')

Further copy the folder **data/demographics** to the data folder in synthpops.