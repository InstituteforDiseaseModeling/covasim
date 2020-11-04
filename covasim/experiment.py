import covasim as cv

import sciris as sc

standardPars = pars = sc.objdict(
    pop_size        = 40000,
    pop_infected    = 10,
    pop_type        = 'synthpops',
    location        = 'Vorarlberg',
    n_days          = 180,
    verbose         = 1,
    pop_scale       = 10,
    n_beds_hosp     = 700 ,  #source: http://www.kaz.bmg.gv.at/fileadmin/user_upload/Betten/1_T_Betten_SBETT.pdf (2019)
    n_beds_icu      = 30,      # source: https://vbgv1.orf.at/stories/493214 (2011 no recent data found)
    iso_factor      = dict(h=1, s=1, w=1, c=1)
)
def plot_res(scenarios):
    fig1 = scenarios.plot(do_show=True, to_plot=sc.odict(

        {
            'cumulativ diagnoses':['cum_diagnoses'],
            'new diagnoses': ['new_diagnoses'],
            'deaths': ['cum_deaths']
        }
    ))

def run_experiment(expName = 'stand_name', scenarios = None, pars = None, metapars = None):
    if pars == None:
        pars = standardPars

    if metapars == None:
        metapars = dict(
            n_runs    = 11,
            rand_seed = 1,
            quantiles = {'low':0.1, 'high':0.9},
        )
    if scenarios == None:
        scenarios = {
            'timeline': {
              'name':'Intervention Timeline in Austria',
              'pars': {
                  'interventions': [
                        #Basistestrate
                        cv.test_num(daily_tests=250,symp_test=100,quar_test=0.9,quar_policy='start',test_delay=2),
                        cv.contact_tracing(trace_probs=0.5,trace_time=2),]
                  }
              },
            }
    vlbgSimulation = cv.sim.Sim(pars=pars, load_pop=True, popfile='voriPop.pop')
   
    scens = cv.Scenarios(sim=vlbgSimulation, metapars=metapars, scenarios=scenarios)

    scens.run(verbose=1)
    
    plot_res(scens)