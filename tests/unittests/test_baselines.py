"""
Tests of simulation parameters from
../../covasim/README.md
"""

import os
import unittest
import difflib
import json
import sciris as sc
import covasim as cv

from unittest_support_classes import CovaSimTest, TestProperties

class BaselineTests(CovaSimTest):
    def setUp(self):
        super().setUp()
        pass

    def tearDown(self):
        super().tearDown()
        pass

    def assertBaseline(self, name, sim):
      # Setup to compare against the existing model 
        path = os.path.join(os.path.dirname(__file__), "../baselines/" + name + ".json")
        current_baseline = None
        new_baseline = None
        updateBaseline = os.environ.get('UPDATE_BASELINE') == '1'

        # Gather the summary as a nice JSON file for easy diffing
        summary = sim.summary_stats(verbose=0)
        new_baseline = json.dumps(summary, indent=2, separators=(',', ': '))

        # If the baseline exists, read it. Otherwise, we'll make it
        if os.path.exists(path):
          current_baseline = open(path, "r").read()
        else:
          updateBaseline = True

        # Update for the user, if they ask or if the baseline did not exist
        if updateBaseline:
          baseline_file = open(path, "w")
          baseline_file.write(new_baseline)
          baseline_file.close()
          current_baseline = new_baseline

        # Assert the baselines are equal with a nice error message if not :)
        # We aren't using assertEqual because that messes with the output whereas we want a very clean output for people to review
        # to achieve the nice output, explicitly fail.
        if current_baseline != new_baseline:
          diff = difflib.unified_diff(current_baseline.splitlines(True), new_baseline.splitlines(True))
          self.fail("""
The baseline has diverged from the expected value. Please review the changes.
If these seem accurate to you, please run `UPDATE_BASELINE=1 pytest {file}`

Difference:
          {diff}
          """.format(diff=''.join(diff), new_baseline=new_baseline, file=__file__))

    def test_baseline_without_intervention(self):
        pars = sc.objdict(
          pop_size     = 20000, # Population size
          pop_infected = 1,     # Number of initial infections
          n_days       = 180,   # Number of days to simulate
          rand_seed    = 1,     # Random seed
        )
        sim = cv.Sim(pars=pars)
        sim.run(verbose=0)
        self.assertBaseline("baseline_without_intervention", sim)

    def test_baseline_with_intervention(self):
      pars = sc.objdict(
        pop_size     = 20000, # Population size
        pop_infected = 1,     # Number of initial infections
        n_days       = 180,   # Number of days to simulate
        rand_seed    = 1,     # Random seed
      )
      pars.interventions = cv.change_beta(days=45, changes=0.5) # Add intervention

      sim = cv.Sim(pars=pars)
      sim.run(verbose=0)
      self.assertBaseline("baseline_with_intervention", sim)
        
