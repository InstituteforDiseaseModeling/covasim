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

    def test_baseline_without_intervation(self):
        # Run the base simulation
        pars = sc.objdict(
            n           = 20000,  # Population size
            n_infected  = 1,      # Number of initial infections
            n_days      = 180,    # Number of days to simulate
            prog_by_age = 1,      # Use age-specific mortality etc.
            usepopdata  = 0,      # Use realistic population structure (requires synthpops)
            seed        = 1,      # Random seed
        )
        sim = cv.Sim(pars=pars)
        sim.run(verbose=0)

        # Setup to compare against the existing model 
        path = os.path.join(os.path.dirname(__file__), "../baselines/run_sim.json")
        current_baseline_file = open(path, "r+")
        current_baseline = current_baseline_file.read()
        summary = sim.summary_stats(verbose=0)
        new_baseline = json.dumps(summary, indent=2, separators=(',', ': '))

        # Update for the user, if they ask
        if os.environ.get('UPDATE_BASELINE') == '1':
          current_baseline_file.write(new_baseline)
          current_baseline_file.close()
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
