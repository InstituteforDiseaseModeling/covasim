'''
Calculate vaccine efficiency for protection against symptomatic covid after first dose
'''

import covasim as cv
import numpy as np
cv.check_version('>=3.0.0')

# construct analyzer to select placebo arm
class placebo_arm(cv.Analyzer):
    def __init__(self, day, trial_size, **kwargs):
        super().__init__(**kwargs)
        self.day = day
        self.trial_size = trial_size
        return

    def initialize(self, sim=None):
        self.placebo_inds = []
        self.initialized = True
        return

    def apply(self, sim):
        if sim.t == self.day:
            eligible = cv.true(~np.isfinite(sim.people.date_exposed) & ~sim.people.vaccinated)
            self.placebo_inds = eligible[cv.choose(len(eligible), min(self.trial_size, len(eligible)))]
        return

pars = {
    'pop_size': 20000,
    'beta': 0.015,
    'n_days': 120,
}

# Define vaccine arm
trial_size = 500
start_trial = 20
def subtarget(sim):
    # select people who are susceptible
    if sim.t == start_trial:
        eligible = cv.true(~np.isfinite(sim.people.date_exposed))
        inds = eligible[cv.choose(len(eligible), min(trial_size//2, len(eligible)))]
    else:
        inds = []
    return {'vals': [1.0 for ind in inds], 'inds': inds}

pfizer = cv.vaccinate(vaccine='pfizer', days=[start_trial], prob=0.0, subtarget=subtarget)

sim = cv.Sim(
    use_waning=True,
    pars=pars,
    interventions=pfizer,
    analyzers=placebo_arm(day=start_trial, trial_size=trial_size//2)
)
sim.run()

# Find trial arm indices, those who were vaccinated
vacc_inds = cv.true(sim.people.vaccinated)
placebo_inds = sim['analyzers'][0].placebo_inds
# Check that there is no overlap
assert (len(set(vacc_inds).intersection(set(placebo_inds))) == 0)
# Calculate vaccine efficiency
VE = 1 - (np.isfinite(sim.people.date_symptomatic[vacc_inds]).sum() /
          np.isfinite(sim.people.date_symptomatic[placebo_inds]).sum())
print('Vaccine efficiency for symptomatic covid:', VE)

# Plot
to_plot = cv.get_default_plots('default', 'sim')
to_plot['Health outcomes'] += ['cum_vaccinated']
sim.plot(to_plot=to_plot)

print('Done')