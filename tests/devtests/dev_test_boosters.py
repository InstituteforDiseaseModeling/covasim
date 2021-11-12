import numpy as np
import sciris as sc
assert sc.__version__ > '1.2.0'
import covasim as cv
import pylab as pl
 
if __name__ == '__main__':

    # Set up a sim
    n_doses = []
    pars = {
        'n_agents': 50_000,
        'beta': 0.015,
        'n_days': 180,
    }

    # Define sequence based vaccination
    def prioritize_by_age(people):
        return np.argsort(-people.age)

    # Start vaccinating 100 people/day on day 30; scale up by 100 doses/day til 65% have had a first dose
    def num_doses(sim):
        if sim.t < 30:
            return 0
        elif sim.t < 50: # First doses only
            return (sim.t - 29) * 100
        elif sim.t < 55: # First and second doses
            return (sim.t-29) * 100 + (sim.t-50) * 100
        elif sim.t < 76: # Second doses only
            return (sim.t - 50) * 100
        else:
            return 0

    # Now give 100 boosters a day starting from day 120
    def num_boosters(sim):
        if sim.t < 120:
            return 0
        else:
            return 100

    # Only give boosters to people who have had 2 doses
    booster_target = {'inds': lambda sim: cv.true(sim.people.vaccinations != 2), 'vals': 0}

    pfizer = cv.vaccinate_num(vaccine='pfizer', sequence=prioritize_by_age, num_doses=num_doses)
    booster = cv.vaccinate_num(vaccine='booster', sequence=prioritize_by_age, subtarget=booster_target, num_doses=num_boosters, booster=True)

    sim = cv.Sim(
        use_waning=True,
        pars=pars,
        interventions=[pfizer, booster],
        analyzers=lambda sim: n_doses.append(sim.people.vaccinations.copy())
    )
    sim.run()

    pl.figure()
    n_doses = np.array(n_doses)
    fully_vaccinated = (n_doses == 2).sum(axis=1)
    first_dose = (n_doses == 1).sum(axis=1)
    boosted = (n_doses > 2).sum(axis=1)
    pl.stackplot(sim.tvec, first_dose, fully_vaccinated, boosted)
    pl.legend(['First dose', 'Fully vaccinated', 'Boosted']);
    pl.show()

