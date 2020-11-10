import covasim as cv
import sciris as sc


pars = sc.objdict(
    pop_size     = 40e3,    # Population size
    location = "Vorarlberg",
    pop_infected = 30,       # Number of initial infections
    start_day = '2020-03-01',
    n_days       = 80,       # Number of days to simulate
    pop_scale = 10,
    n_beds_icu = 30,
    n_beds_hosp = 700,
    #contacts = 0.5
)

# Scenario metaparameters
metapars = dict(
    n_runs    = 11, # Number of parallel runs; change to 3 for quick, 11 for real
    noise     = 0.0, # Use noise, optionally
    noisepar  = 'beta',
    rand_seed = 1,
    #quantiles = {'low':0.1, 'high':0.9},
)

# Define the actual scenarios
lockdown1_date = '2020-03-16'
lockdown1_store_opening_date = '2020-04-14'
lockdown1_school_opening_date = '2020-05-01'
lockdown1_restaurant_opening_date = '2020-05-15'
reduced_masks_date = '2020-06-15'
increased_testing_date = '2020-'


scenarios = {
            'timeline': {
              'name':'Intervention Timeline in Austria',
              'pars': {
                  'interventions': [
                        #Basistestrate
                        cv.test_num(daily_tests=250,symp_test=100,quar_test=0.9,quar_policy='start',test_delay=2),
                        cv.contact_tracing(trace_probs=0.5,trace_time=2),
                        ##Lockdown 16.03
                        cv.clip_edges(days=lockdown1_date,changes=0.6,layers='w'),
                        cv.clip_edges(days=lockdown1_date,changes=0.15,layers='s'),
                        cv.clip_edges(days=lockdown1_date,changes=0.4,layers='c'),
                        cv.change_beta(days=lockdown1_date,changes=0.7),

                        ## Eröffnung Geschaefte
                        cv.clip_edges(days=lockdown1_store_opening_date,changes=0.5,layers='c'),

                        ##Eroeffnung Schulen
                        cv.clip_edges(days=lockdown1_school_opening_date,changes=0.8,layers='s'),

                        ##Eroeffnung Gastronomie
                        cv.clip_edges(days=lockdown1_restaurant_opening_date,changes=0.5,layers='c'),

                        ##Reduktion Maskenpflicht
                        cv.change_beta(days=reduced_masks_date,changes=0.9)
                        ]
                  }
              },
            }


if __name__ == "__main__":
    mysim = cv.sim.Sim(pars=pars, load_pop=True, popfile='vlbgPop40000.pop')
    metapars = dict(
            n_runs    = 11,
            rand_seed = 1,
            quantiles = {'low':0.1, 'high':0.9},
        )
    scens = cv.Scenarios(sim=mysim, metapars=metapars, scenarios=scenarios)
    #myMulti = cv.MultiSim(sims=scens,n_runs=11,noise=0.0,keep_People = True, quantiles={'low':0.1,'high':0.9})
    scens.run(verbose=1)

    scens.plot()
    
    
#multiSim = cv.MultiSim(sims=mysim,n_runs=2)
#mysim.run()
#mysim.plot()
