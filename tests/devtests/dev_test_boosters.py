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

    # Start vaccinating 100 people/day on day 30; scale up by 100 doses/day til 65% are vaccinated
    def num_doses(sim):
        if sim.t < 30:
            return 0
        elif sim.t < 50: # First doses only
            return (sim.t - 29) * 100
        elif sim.t < 55: # First and second doses
            return (sim.t-29) * 100 + (sim.t-21-29) * 100
        elif sim.t < 76: # Second doses only
            return (sim.t - 50) * 100
        else:
            return 0

    pfizer = cv.vaccinate_num(vaccine='pfizer', sequence=prioritize_by_age, num_doses=num_doses)
    sim = cv.Sim(
        use_waning=True,
        pars=pars,
        interventions=pfizer,
        analyzers=lambda sim: n_doses.append(sim.people.vaccinations.copy())
    )
    sim.run()

    pl.figure()
    n_doses = np.array(n_doses)
    fully_vaccinated = (n_doses == 2).sum(axis=1)
    first_dose = (n_doses == 1).sum(axis=1)
    pl.stackplot(sim.tvec, first_dose, fully_vaccinated)
    pl.legend(['First dose', 'Fully vaccinated']);
    pl.show()

    # # Proposed new syntax
    # # Must accommodate different dosing schedules, boosters, and mixed-vaccines
    #
    # # Example
    # # Phase 1: prioritize >80
    # phase1 = cv.vaccinate_num(vaccine='pfizer', sequence=prioritize_by_age, num_doses=)
    #
    # pfizer = cv.vaccinate_num(vaccine='pfizer', sequence=prioritize_by_age, num_doses=num_doses)
    # sim = cv.Sim(
    #     use_waning=True,
    #     pars=pars,
    #     interventions=pfizer,
    #     analyzers=lambda sim: n_doses.append(sim.people.vaccinations.copy())
    # )
    # sim.run()
    #
    #
    # def age_sequence(people): return np.argsort(-people.age)
    # def num_boosters(sim):
    #     if sim.t < 100: return 0
    #     else: return 100
    # pfizer = cv.vaccinate_num(vaccine='pfizer', sequence=age_sequence, num_doses=100)
    # pfizer_boost = cv.vaccinate_num(vaccine='pfizer_boost', sequence=age_sequence, num_doses=num_boosters)
    # cv.Sim(interventions=[pfizer, pfizer_boost], use_waning=True)
