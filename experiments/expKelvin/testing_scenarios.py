import covasim as cv
import sciris as sc

# Run options
do_plot = 1
do_show = 1
verbose = 1
start_day = '2020-04-04'

pars = pars = sc.objdict(
    pop_size        = 40000,
    pop_infected    = 10,
    pop_type        = 'synthpops',
    location        = 'Vorarlberg',
    n_days          = 180,
    verbose         = 1,
    pop_scale       = 10,
    n_beds_hosp     = 700 ,  #source: http://www.kaz.bmg.gv.at/fileadmin/user_upload/Betten/1_T_Betten_SBETT.pdf (2019)
    n_beds_icu      = 30,      # source: https://vbgv1.orf.at/stories/493214 (2011 no recent data found)
    iso_factor      = dict(h=0.3, s=0.1, w=0.1, c=0.1), # change this afterwards; default: 0.3, 0.1, 0.1, 0.1
    #quar_factor = dict(h=0.8, s=0.0, w=0.0, c=0.1),
    start_day       = start_day
)

# Scenario metaparameters
metapars = dict(
    n_runs    = 2, # Number of parallel runs; change to 3 for quick, 11 for real
    noise     = 0.1, # Use noise, optionally
    noisepar  = 'beta',
    rand_seed = 1,
    quantiles = {'low':0.4, 'high':0.6},
)

sequenceIntervention = [
                            cv.test_prob(symp_prob=0.1, asymp_prob=0.002),
                            cv.test_prob(symp_prob=0.3, asymp_prob=0.1),
                            cv.test_prob(symp_prob=0.95, asymp_prob=0.8)
                        ]

sim = cv.Sim(pars)
sim.init_people(load_pop=True, popfile='voriPop.pop')


scenarios = { 'day_one': {
              'name':'Day One Testing',
              'pars': {
                  'interventions': [
                      cv.test_prob(start_day=start_day, test_sensitivity=0.98, test_delay=1, symp_prob=1, asymp_prob=1)
                  ]
                  }
            },
            'sequence0': {
              'name':'sequence0 - testing',
              'pars': {
                  'interventions': [
                        cv.sequence(days=[25, 50, 75], interventions=sequenceIntervention)
                    ]
                  }
            },
            'sequence1': {
            'name':'sequence1 - s/w beta change',
            'pars': {
                'interventions': [
                    cv.sequence(days=[25, 50, 75], interventions=sequenceIntervention),
                    cv.change_beta(days=[25, 50, 75], changes=[0.85, 0.8, 0.2], layers='s'),
                    cv.change_beta(days=[25, 50, 75], changes=[0.8, 0.5, 0.2], layers='w')
                ]
                }
            },
            'sequence2': {
            'name':'sequence2 - beta dynamic pars',
            'pars': {
                'interventions': [
                    cv.sequence(days=[25, 50, 75], interventions=sequenceIntervention),
                    cv.dynamic_pars({'beta':{'days':[25, 50, 75], 'vals':[0.025, 0.015, 0.005]}}) # this is not working?
                ]
                }
            },
            'sequence3': {
            'name':'sequence3 - age55, quarantine',
            'pars': {
                'interventions': [
                    cv.sequence(days=[25, 50, 75], interventions=[
                            cv.test_prob(symp_prob=0.1, asymp_prob=0.002, subtarget={'inds': sim.people.age>55, 'vals': 1.5}),
                            cv.test_prob(symp_prob=0.3, asymp_prob=0.1, subtarget={'inds': sim.people.age>55, 'vals': 1.5}),
                            cv.test_prob(symp_prob=0.95, asymp_prob=0.8, subtarget={'inds': sim.people.age>55, 'vals': 1.5})
                        ],
                    ),
                    cv.sequence(days=[25, 50, 75], interventions=sequenceIntervention),
                    cv.contact_tracing(trace_probs=0.4, trace_time=3) # quar_period not available
                ]
                }
            },
        }

sim = cv.Sim(pars)
sim.init_people(load_pop=True, popfile='voriPop.pop')

if __name__ == '__main__':
    scens = cv.Scenarios(sim=sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose)
    if do_plot:
        fig1 = scens.plot(do_show=do_show)