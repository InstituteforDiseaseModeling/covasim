import covasim as cova
import unittest


class TestProperties:
    Pars_Test_list = ["apple", "banana", "coconut"]
    Pars_Test_string = "Mega Ran"
    Pars_Test_func = str.split
    Pars_Test_int = 42
    Pars_Test_float = 3.14159
    Pars_Test_bool = False
    class Person:
        hat = "Cowboy"
        movie = "Paddington II"
        dog = "Corgi"
        pi_approximation = 3.14159
        pass
    class Sim:
        expected_number_people = 123
        expected_number_days = 365
        expected_disease = "rhinovirus"
        expected_seed = 112233

class DummySim(cova.Sim):
    def __init__(self):
        sim_parameters = {
            "num_people" : TestProperties.Sim.expected_number_people,
            "num_days" : TestProperties.Sim.expected_number_days,
            "disease_selection" : TestProperties.Sim.expected_disease
        }
        super().__init__(sim_parameters)
        self.people = {}
        self.uids = {}
        pass

    def init_people(self):
        ages_in_question = [1, 5, 10, 20, 21, 40, 41, 42, 60, 70, 80]
        age_length = len(ages_in_question)
        for p in range(1, self['num_people'] + 1):
            sex = p % 2
            age_index = p % age_length
            person_params = {
                "age": ages_in_question[age_index],
                "sex": sex
            }
            local_person = cova.Person(person_params)
            self.uids[p] = 1000 + p
            self.people[self.uids[p]] = local_person

class CovaUnitTests(unittest.TestCase):

    def setUp(self):
        self.pars_obj = None
        self.person = None
        self.sim = None
        pass

    def tearDown(self):
        pass

    # region parsobj
    def create_predictable_parsobj(self, parameters_dict=None):
        if not parameters_dict:
            parameters = {
                "test_list": TestProperties.Pars_Test_list,
                "test_string": TestProperties.Pars_Test_string,
                "test_func": TestProperties.Pars_Test_func,
                "test_int": TestProperties.Pars_Test_int,
                "test_float": TestProperties.Pars_Test_float,
                "test_bool": TestProperties.Pars_Test_bool
            }
        else:
            parameters = parameters_dict
        self.pars_obj = cova.ParsObj(pars=parameters)

    def test_parsobj_init_get(self):
        self.create_predictable_parsobj()
        self.assertIsNotNone(self.pars_obj)
        self.assertEqual(TestProperties.Pars_Test_bool, self.pars_obj["test_bool"])
        self.assertEqual(TestProperties.Pars_Test_float, self.pars_obj["test_float"])
        self.assertEqual(TestProperties.Pars_Test_func, self.pars_obj["test_func"])
        self.assertEqual(TestProperties.Pars_Test_int, self.pars_obj["test_int"])
        self.assertEqual(TestProperties.Pars_Test_list, self.pars_obj["test_list"])
        self.assertEqual(TestProperties.Pars_Test_string, self.pars_obj["test_string"])
        pass

    def test_parsobj_func(self):
        self.create_predictable_parsobj()
        expected_array = ["Mega","Ran"]
        actual_array = self.pars_obj["test_func"](TestProperties.Pars_Test_string)
        self.assertEqual(expected_array, actual_array)
        pass

    def test_parsobj_set(self):
        self.create_predictable_parsobj()

        expected_string = "Random"
        expected_list = [1, 9, 2, 8, 3, 4, 5]
        expected_func = str.join
        expected_int = 255
        expected_float = 1.61803
        expected_bool = True

        self.pars_obj["test_string"] = expected_string
        self.pars_obj["test_list"] = expected_list
        self.pars_obj["test_func"] = expected_func
        self.pars_obj["test_int"] = expected_int
        self.pars_obj["test_float"] = expected_float
        self.pars_obj["test_bool"] = expected_bool

        self.assertEqual(expected_bool, self.pars_obj["test_bool"])
        self.assertEqual(expected_float, self.pars_obj["test_float"])
        self.assertEqual(expected_func, self.pars_obj["test_func"])
        self.assertEqual(expected_int, self.pars_obj["test_int"])
        self.assertEqual(expected_list, self.pars_obj["test_list"])
        self.assertEqual(expected_string, self.pars_obj["test_string"])
        pass

    def test_parsobj_set_newkey(self):
        self.create_predictable_parsobj()
        surprise_key = "test_surprise"
        expected_surprise = "its an egg"
        with self.assertRaises(KeyError) as context:
            self.pars_obj[surprise_key] = expected_surprise
            pass
        exception_message = str(context.exception)
        self.assertIn(surprise_key, exception_message)

    def test_update_pars(self):
        self.create_predictable_parsobj()

        test_parameters = {
            "fruits": ["apples", "bananas", "coconuts"],
            "dogs": ["Akita", "Beagle", "Corgi"],
            "name": "Jeff Boogaloo",
            1 : "Megaman"
        }
        self.pars_obj.update_pars(test_parameters)
        test_parameters_keys = list(test_parameters.keys())
        # pars_obj_keys = list(self.pars_obj.keys()) # TODO: add a keys() method?
        # self.assertEqual(test_parameters_keys, pars_obj_keys)
        for k in test_parameters_keys:
            self.assertEqual(test_parameters[k], self.pars_obj[k])
        pass
    # endregion

    # region Person
    def create_predictable_person_pars(self):
        parameters = {
            "favorite_hat": TestProperties.Person.hat,
            "favorite_dog": TestProperties.Person.dog,
            "favorite_movie": TestProperties.Person.movie,
            "best_pi_approximation": TestProperties.Person.pi_approximation
        }
        self.person = cova.Person(pars=parameters)
        pass

    def test_person_init(self):
        self.assertIsNone(self.person)
        self.create_predictable_person_pars()
        self.assertIsNotNone(self.person)
        pass
    # endregion

    # region Sim
    def create_predicatable_sim(self):
        parameters_dict = {
            "num_people" : TestProperties.Sim.expected_number_people,
            "number_days" : TestProperties.Sim.expected_number_days,
            "disease_to_simulate" : TestProperties.Sim.expected_disease,
            "seed": TestProperties.Sim.expected_seed
        }
        self.sim = cova.Sim(pars=parameters_dict)
        pass

    def create_dummy_sim(self):
        self.sim = DummySim()
        self.sim.init_people()

    def test_sim_init(self):
        self.assertIsNone(self.sim)
        self.create_predicatable_sim()
        self.assertIsNotNone(self.sim)
        pass

    @unittest.skip("The set seed method is talking to numpy and skips the sim object")
    def test_sim_set_seed_noreset(self):
        self.create_predicatable_sim()
        expected_seed = 332211
        self.sim.set_seed(seed=expected_seed)
        actual_seed = "BUGBUG: no way to check this" # np.random.get_state[0][1]
        self.assertEqual(expected_seed, actual_seed)
        pass

    @unittest.skip("The set seed method is talking to numpy and skips the sim object")
    def test_sim_set_seed_reset(self):
        self.create_predicatable_sim()
        expected_seed = 332211
        self.sim.set_seed(seed=expected_seed, reset=True)
        actual_seed = "BUGBUG: no way to check this" # np.random.get_state[0][1]
        self.assertEqual(expected_seed, actual_seed)
        pass

    def test_sim_n_people(self):
        self.create_dummy_sim()
        people_count = self.sim.n
        self.assertEqual(TestProperties.Sim.expected_number_people, people_count)
        pass

    @unittest.skip("NYI on test side")
    def test_sim_n_points(self):
        pass

    @unittest.skip("NYI on test side")
    def test_sim_time_vector(self):
        pass

    def test_sim_get_person(self):
        self.create_dummy_sim()
        expected_sex_1 = 1
        expected_age_1 = 5
        expected_sex_2 = 0
        expected_age_2 = 10
        p1 = self.sim.get_person(1)
        self.assertIsNotNone(p1)
        p2 = self.sim.get_person(2)
        self.assertIsNotNone(p2)
        self.assertEqual(expected_age_1, p1["age"])
        self.assertEqual(expected_sex_1, p1["sex"])
        self.assertEqual(expected_age_2, p2["age"])
        self.assertEqual(expected_sex_2, p2["sex"])
        pass

    def test_sim_NYI_methods(self):
        pass
    # endregion