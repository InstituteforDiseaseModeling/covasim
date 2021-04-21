'''
Contains code that allows for customized test sensitivity profiles to be used, and
more custom testing behaviour.
'''

import covasim as cv
import numpy as np
from . import utils as cvu

class TestSensitivityProfile:

    def __init__(self):
        return

    def initialize(self, people):
        """Initialize the test sensitivity profiles

        Args:
            people (cv.people): The people object of the current simulation
        """
        self.initialized = True
        return

    def test(self, people, inds):
        """
        Method to test people, and update the cv.people object.
        """
        return

class BinomialTestSensitivityProfile(TestSensitivityProfile):

    def __init__(self, test_sensitivity: float, loss_prob: float, test_delay: int):
        """Uses a simple model of testing, where there is a fixed delay
        to receive test results. If the individual is still infected, then there 
        is a fixed probability of testing positive. Some individuals who test positive
        will be lost to follow up.

        Args:
            test_sensitivity ([type]): [description]
            loss_prob ([type]): [description]
            test_delay ([type]): [description]
        """
        self.test_sensitivity = test_sensitivity
        self.loss_prob        = loss_prob
        self.test_delay       = test_delay

    def test(self, people, inds):
        '''
        Method to test people. Typically not to be called by the user directly;
        see the test_num() and test_prob() interventions.

        Args:
            inds: indices of who to test
            test_sensitivity (float): probability of a true positive
            loss_prob (float): probability of loss to follow-up
            test_delay (int): number of days before test results are ready
        '''

        # extract useful quantities from the simulation
        t = people.t

        inds = np.unique(inds)
        people.tested[inds] = True
        people.date_tested[inds] = t # Only keep the last time they tested

        is_infectious = cvu.itruei(people.infectious, inds)
        pos_test      = cvu.n_binomial(self.test_sensitivity, len(is_infectious))
        is_inf_pos    = is_infectious[pos_test]

        not_diagnosed = is_inf_pos[np.isnan(self.date_diagnosed[is_inf_pos])]
        not_lost      = cvu.n_binomial(1.0-self.loss_prob, len(not_diagnosed))
        final_inds    = not_diagnosed[not_lost]

        # Store the date the person will be diagnosed, as well as the date they took the test which will come back positive
        self.date_diagnosed[final_inds] = t + self.test_delay
        self.date_pos_test[final_inds] = t
