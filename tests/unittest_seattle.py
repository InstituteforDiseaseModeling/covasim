from covasim.cova_seattle import Person
from covasim.cova_seattle import parameters as seattle_parameters
from covasim.cova_seattle import Sim
import unittest
import numpy as np

class CovaSeattleUnittests(unittest.TestCase):

    def setUp(self):
        self.bonnie = None
        self.clyde = None
        self.seattle_pars = None
        self.seattle_sim = None
        pass

    def tearDown(self):
        pass

    # region Seattle Helpers
    def create_predictable_person_pars(self):
        self.bonnie = Person(pars={}, age=24, sex=0)
        self.clyde = Person(pars={}, age=25, sex=1)
        self.people = [self.bonnie, self.clyde]
        pass

    def create_seattle_parameters(self):
        self.seattle_pars = seattle_parameters.make_pars()
        pass

    def create_seattle_sim_default(self):
        self.create_seattle_parameters()
        self.seattle_sim = Sim(self.seattle_pars)
    # endregion

    # region SeattlePerson
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

    # region SeattleParameters
    def test_seattle_pars_created(self):
        self.assertIsNone(self.seattle_pars)
        self.create_seattle_parameters()
        self.assertIsNotNone(self.seattle_pars)
        self.assertEqual(type(self.seattle_pars), dict)
        self.assertEqual(self.seattle_pars["seed"], 1)
        pass
    # endregion

    # region SeattleSim
    def test_sim_init_default_parameters(self):
        self.assertIsNone(self.seattle_sim)
        self.create_seattle_sim_default()
        self.assertIsNotNone(self.seattle_sim)

        # test init_results
        self.assertFalse(self.seattle_sim.results['ready'])
        results_keys = self.seattle_sim.results_keys
        for k in results_keys:
            if "infections" != k: # We start with 4 of those right now
                total_channel = np.sum(self.seattle_sim.results[k])
                self.assertEqual(total_channel, 0, msg=f"Channel {k} should equal zero. Got {total_channel}\n")
            pass
        self.assertEqual({}, self.seattle_sim.results['transtree'])

        # test init_people
        people = self.seattle_sim.people
        self.assertEqual(self.seattle_pars["n"], len(people))

        # test init interventions
        self.assertEqual({}, self.seattle_sim.interventions)
        pass

    def test_sim_infect_person(self):
        self.create_seattle_sim_default()
        infector_uuid = self.seattle_sim.uids[0]
        infector = self.seattle_sim.people[infector_uuid]

        victim_uuid = self.seattle_sim.uids[1]
        victim = self.seattle_sim.people[victim_uuid]

        infection_time = 42

        target_person = self.seattle_sim.infect_person(infector, victim, infection_time)
        self.assertFalse(victim.susceptible)
        self.assertTrue(victim.exposed)
        self.assertEqual(victim.date_exposed, infection_time)

        transmission_tree = self.seattle_sim.results['transtree']
        self.assertIn(victim_uuid, transmission_tree) # victim now in transmission tree

        transmission_event = transmission_tree[victim_uuid]
        self.assertEqual(transmission_event['from'], infector_uuid)
        self.assertEqual(transmission_event['date'], infection_time)
    # endregion
    pass
