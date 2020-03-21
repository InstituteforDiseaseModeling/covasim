from covasim import Person
from covasim import parameters
from covasim import Sim
import unittest
import numpy as np

class CovagenericUnittests(unittest.TestCase):

    def setUp(self):
        self.bonnie = None
        self.clyde = None
        self.generic_pars = None
        self.generic_sim = None
        pass

    def tearDown(self):
        pass

    # region generic Helpers
    def create_predictable_person_pars(self):
        self.bonnie = Person(pars={}, age=24, sex=0)
        self.clyde = Person(pars={}, age=25, sex=1)
        self.people = [self.bonnie, self.clyde]
        pass

    def create_parameters(self):
        self.generic_pars = parameters.make_pars()
        pass

    def create_generic_sim_default(self):
        self.create_parameters()
        self.generic_sim = Sim(self.generic_pars)
    # endregion

    # region genericPerson
    def test_person_init(self):
        self.create_predictable_person_pars()
        for person in self.people:
            self.assertTrue(person.alive)
            self.assertTrue(person.susceptible)
            self.assertFalse(person.exposed)
            self.assertFalse(person.infectious)
            self.assertFalse(person.diagnosed)
            self.assertFalse(person.recovered)
            self.assertFalse(person.dead)
        self.assertEqual(self.bonnie.age, 24)
        self.assertEqual(self.bonnie.sex, 0)
        self.assertEqual(self.clyde.age, 25)
        self.assertEqual(self.clyde.sex, 1)
        self.assertNotEqual(self.bonnie.uid, self.clyde.uid)
    # endregion

    # region genericParameters
    def test_generic_pars_created(self):
        self.assertIsNone(self.generic_pars)
        self.create_parameters()
        self.assertIsNotNone(self.generic_pars)
        self.assertEqual(type(self.generic_pars), dict)
        self.assertEqual(self.generic_pars["seed"], 1)
        pass
    # endregion

    # region genericSim
    def test_sim_init_default_parameters(self):
        self.assertIsNone(self.generic_sim)
        self.create_generic_sim_default()
        self.assertIsNotNone(self.generic_sim)

        # test init_results
        self.assertFalse(self.generic_sim.results['ready'])
        reskeys = self.generic_sim.reskeys
        for k in reskeys:
            if "infections" != k: # We start with 4 of those right now
                total_channel = np.sum(self.generic_sim.results[k].values)
                self.assertEqual(total_channel, 0, msg=f"Channel {k} should equal zero. Got {total_channel}\n")
            pass
        self.assertEqual({}, self.generic_sim.results['transtree'])

        # test init_people
        people = self.generic_sim.people
        self.assertEqual(self.generic_pars["n"], len(people))

        # test init interventions
        self.assertEqual({}, self.generic_sim.interventions)
        pass

    def test_sim_infect_person(self):
        self.create_generic_sim_default()
        infector_uuid = self.generic_sim.uids[0]
        infector = self.generic_sim.people[infector_uuid]

        victim_uuid = self.generic_sim.uids[1]
        victim = self.generic_sim.people[victim_uuid]

        infection_time = 42

        target_person = self.generic_sim.infect_person(infector, victim, infection_time)
        self.assertFalse(victim.susceptible)
        self.assertTrue(victim.exposed)
        self.assertEqual(victim.date_exposed, infection_time)

        transmission_tree = self.generic_sim.results['transtree']
        self.assertIn(victim_uuid, transmission_tree) # victim now in transmission tree

        transmission_event = transmission_tree[victim_uuid]
        self.assertEqual(transmission_event['from'], infector_uuid)
        self.assertEqual(transmission_event['date'], infection_time)
    # endregion
    pass
