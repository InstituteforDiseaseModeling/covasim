import covasim as cv
import sciris as sc


pars = sc.objdict(
    pop_size     = 40e3,    # Population size
    location = "Vorarlberg",
    pop_infected = 10,       # Number of initial infections
    n_days       = 150,       # Number of days to simulate
    pop_scale = 10,
    n_beds_icu = 200,
    n_beds_hosp = 400,
    #contacts = 0.5
)

# Scenario metaparameters
metapars = dict(
    n_runs    = 3, # Number of parallel runs; change to 3 for quick, 11 for real
    noise     = 0.1, # Use noise, optionally
    noisepar  = 'beta',
    rand_seed = 1,
    quantiles = {'low':0.1, 'high':0.9},
)

# Define the actual scenarios
start_day = '2020-04-01'
start_testing_day = '2020-04-21'
scenarios = {'baseline': {
              'name':'Baseline',
              'pars': {
                  'interventions': None,
                  }
              },
            'distance': {
              'name':'Social distancing',
              'pars': {
                  'interventions': cv.change_beta(days=start_day, changes=0.7)
                  }
              },
            'ttq': {
              'name':'Test-trace-quarantine',
              'pars': {
                  'interventions': [
                        cv.test_prob(start_day=start_day, symp_prob=0.2, asymp_prob=0.05, test_delay=1.0),
                        cv.contact_tracing(start_day=start_day, trace_probs=0.8, trace_time=1.0),
                    ]
                  }
              },
            'ttq20': {
            'name':'Test-trace-quarantine 20',
            'pars': {
                'interventions': [
                    cv.test_prob(start_day=start_testing_day, symp_prob=0.2, asymp_prob=0.05, test_delay=1.0),
                    cv.contact_tracing(start_day=start_testing_day, trace_probs=0.8, trace_time=1.0),
                ]
                }
            },
            }


if __name__ == "__main__":
    mysim = cv.sim.Sim(pars=pars, load_pop=True, popfile='voriPop.pop')
    scens = cv.Scenarios(sim=mysim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=1)
    scens.plot()
#multiSim = cv.MultiSim(sims=mysim,n_runs=2)
#mysim.run()
#mysim.plot()

