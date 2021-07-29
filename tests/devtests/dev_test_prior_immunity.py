import covasim as cv
import numpy as np


def example_estimate_prob():
    import matplotlib.pyplot as plt

    # check single day estimation campaign:
    duration, coverage = (1, 0.3)
    p = cv.historical_vaccinate_prob.estimate_prob(duration=duration, coverage=coverage)
    assert(np.isclose(p, coverage))
    print('single day campaign:', coverage, '\simeq', p)

    # plot campaign coverages as a function of daily probability and campaign duration
    durations = np.arange(1, 60)
    probs = [0.0008, 0.008, 0.08, 0.8]
    plt.figure()
    for ix, prob in enumerate(probs):
        coverage = cv.historical_vaccinate_prob.NB_cdf(durations-1, 1 - prob)
        plt.plot(durations, coverage, label=prob)
    plt.legend()
    plt.xlabel('Duration of Campaign')
    plt.ylabel('Fraction of population vaccinated')
    plt.show()


def example1():
    # length of our base campaign
    duration = 30
    # estimate per-day probability needed for a coverage of 30%
    prob = cv.historical_vaccinate_prob.estimate_prob(duration=duration, coverage=0.30)
    print('using per-day probability of ', prob)

    pfizer = cv.historical_vaccinate_prob(vaccine='pfizer', days=np.arange(-duration, 0), prob=prob)
    sim = cv.Sim(interventions=pfizer, use_waning=True)

    sim.run()

    to_plot = cv.get_default_plots(kind='sim')
    to_plot['Total counts'] += ['cum_vaccinated']
    to_plot['Daily counts'] += ['new_vaccinations']
    sim.plot(to_plot=to_plot)


def example2():
    pars = {'use_waning': True}
    variants = [cv.variant('b117', days=30, n_imports=10)]
    sim = cv.Sim(pars=pars, variants=variants)

    # length of our base campaign
    duration = 30
    # estimate per-day probability needed for a coverage of 30%
    prob = cv.historical_vaccinate_prob.estimate_prob(duration=duration, coverage=0.30)
    print('using per-day probability of ', prob)

    # estimate per-day probability needed for a coverage of 30%
    prob2 = cv.historical_vaccinate_prob.estimate_prob(duration=2*duration, coverage=0.30)

    scenarios = {
        'base':{
            'name': 'baseline',
            'pars': {}
        },
        'scen1':{
            'name': 'historical_vaccinate',
            'pars': {
                'interventions':[cv.historical_vaccinate_prob(vaccine='pfizer', days=np.arange(-duration, 0), prob=prob)]
            }
        },
        'scen2': {
            'name': 'vaccinate',
            'pars': {
                'interventions': [cv.vaccinate_prob(vaccine='pfizer',days=np.arange(0, 30), prob=prob)]
            }
        },
        'scen3': {
            'name': 'historical_vaccinate into sim',
            'pars': {
                'interventions': [cv.historical_vaccinate_prob(vaccine='pfizer', days=np.arange(-30, 30), prob=prob2)]
            }
        },
    }

    scens = cv.Scenarios(sim=sim, scenarios=scenarios)

    scens.run()

    scens.plot()


if __name__ == "__main__":

    # single vaccine campaign example
    example1()

    # compare vaccinate and historical vaccinate
    example2()

    # examples using estimate_prob
    example_estimate_prob()