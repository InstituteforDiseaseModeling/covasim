import covasim as cv

import sciris as sc
import os
standardPars = pars = sc.objdict(
    pop_size        = 40000,
    pop_infected    = 10,
    start_day       = '2020-03-01',
    location        = 'Vorarlberg',
    n_days          = 180,
    verbose         = 1,
    pop_scale       = 10,
    n_beds_hosp     = 700 ,  #source: http://www.kaz.bmg.gv.at/fileadmin/user_upload/Betten/1_T_Betten_SBETT.pdf (2019)
    n_beds_icu      = 30,      # source: https://vbgv1.orf.at/stories/493214 (2011 no recent data found)
    iso_factor      = dict(h=1, s=1, w=1, c=1)
)

def summarize_results(scenarios,summaryPath):
    summaryStr = 'Simulation summary:\n'
    scenario_names = list(scenarios.scenarios.keys())
    for name in scenario_names:
        t = len(scenarios.results[0][name]['best'])-1

        summaryStr+=name+'\n'
        for key in scenarios.result_keys():
                    if key.startswith('cum_'):
                        summaryStr += key+': '+str(round(scenarios.results[key][name]['best'][t],3))+'\n'
                        #summary_str += f'   {summary[key]:5.0f} {key}\n'
        summaryStr+='\n'
        print(summaryStr)
    
    summaryFile = open(summaryPath,'w')
    summaryFile.write(summaryStr)
    summaryFile.close()


def plot_res(scenarios,expName = 'res'):
    notCreated = True
    targetDirectory = os.path.join('Results',expName)
    cnt = 0
    while(notCreated):
        try:
            os.mkdir(targetDirectory+str(cnt))
        except OSError:
            print("CreationFailed")
            cnt+=1
        else:
            notCreated = False
    
    filePath = os.path.join(targetDirectory+str(cnt),expName+'Results.xlsx')
    scenarios.to_excel(filePath)
    summaryPath = os.path.join(targetDirectory+str(cnt),expName+'Summary.txt')
    #summarize_results(scenarios,summaryPath)
    figPath = os.path.join(targetDirectory+str(cnt),expName+'Results.png')
    scenarios.plot(do_show=True,do_save=True,fig_path=figPath, to_plot=sc.odict(

        {
            'cumulativ infections':['cum_infections'],
            'new infections': ['new_infections'],
            'deaths': ['cum_deaths']
        }
    ))

def plot_res_diagnoses(scenarios,expName = 'res'):
    notCreated = True
    targetDirectory = os.path.join('Results',expName)
    cnt = 0
    while(notCreated):
        try:
            os.mkdir(targetDirectory+str(cnt))
        except OSError:
            print("Creation Failed")
            cnt+=1
        else:
            notCreated = False
    
    filePath = os.path.join(targetDirectory+str(cnt),expName+'Results.xlsx')
    scenarios.to_excel(filePath)
    summaryPath = os.path.join(targetDirectory+str(cnt),expName+'Summary.txt')
    summarize_results(scenarios,summaryPath)
    figPath = os.path.join(targetDirectory+str(cnt),expName+'Results.png')
    scenarios.plot(do_show=True,do_save=True,fig_path=figPath, to_plot=sc.odict(

        {
            'cumulativ diagnoses':['cum_diagnoses'],
            'new diagnoses': ['new_diagnoses'],
            'deaths': ['cum_deaths']
        }
    ))


def run_experiment(expName = 'stand_name', scenarios = None, pars = None, metapars = None, do_plot = True):
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
    vlbgSimulation = cv.sim.Sim(pars=pars, load_pop=True, popfile='vlbgPop40000.pop')
   
    scens = cv.Scenarios(sim=vlbgSimulation, metapars=metapars, scenarios=scenarios)

    scens.run(verbose=1)
    if(do_plot==True):
        plot_res(scens,expName=expName)
    return scens