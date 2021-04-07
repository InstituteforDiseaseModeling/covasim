#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import numpy.random as npr
from functools import partial


# In[ ]:


class GetTestSensitivityCurve:

    def __init__(self, path1, path2):
        self.pcr_pars = pd.read_csv(path1, index_col=0)
        self.lfa_pars = pd.read_csv(path2, index_col=0)

        # get parameters from the PCR posterior distribution
        pcr_pars = self.get_pcr_pars()

        # create a pcr sensitivity curve
        self.pcr_prob_positive = partial(
            self.positivity_curve,
            breakpoint_par=pcr_pars.breakpoint_par,
            intercept=pcr_pars.intercept,
            slope_regression_1=pcr_pars.slope_regression_1,
            slope_regression_2=pcr_pars.slope_regression_2
        )

        # get parameters from the LFA posterior distribution
        lfa_pars = self.get_lfa_pars()

        # create a LFA sensitivity curve based on the posterior sample
        self.lfa_prob_positive = partial(
            self.positivity_curve,
            breakpoint_par=lfa_pars.breakpoint_par,
            intercept=lfa_pars.intercept,
            slope_regression_1=lfa_pars.slope_regression_1,
            slope_regression_2=lfa_pars.slope_regression_2
        )

    def step_function(self, infectious_age, breakpoint_par):
        if (infectious_age > breakpoint_par):
            return 1
        else:
            return 0

    def inv_logit(self, x):
        return np.exp(x) / (np.exp(x) + 1)

    # returns a test sensitivity curve from a piecewise logistic regression
    def positivity_curve(self, infectious_age, breakpoint_par, intercept, slope_regression_1, slope_regression_2):
        time_relative_to_breakpoint = (infectious_age - breakpoint_par)
        coefficient = intercept + slope_regression_1 * time_relative_to_breakpoint + slope_regression_1 * slope_regression_2 * time_relative_to_breakpoint * self.step_function(
            infectious_age, breakpoint_par)
        return self.inv_logit(coefficient)

    # sample from the pcr par posterior
    def get_pcr_pars(self):
        index = npr.choice(list(range(1, 4000)))
        return self.pcr_pars.loc[index, :]

    # sample from the lfa par posterior
    def get_lfa_pars(self):
        index = npr.choice(list(range(1, 4000)))
        return self.lfa_pars.loc[index, :]

