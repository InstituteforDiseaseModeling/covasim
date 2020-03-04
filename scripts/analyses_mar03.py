'''
Run analyses
'''

import pylab as pl
import sciris as sc
import covid_abm

run_calibration = False
run_scenarios = True

if run_calibration:
    filename = 'calibration_2020mar03.png'
    verbose = 0
    sim = covid_abm.Sim()
    sim.set_seed(5)
    sim.run(verbose=verbose)
    sim.likelihood()
    sim.plot(do_save=filename)


if run_scenarios:
    filename = 'scenarios_2020mar03.png'
    verbose = 0
    rerun = 0
    
    if rerun:
        # Status quo
        sim0 = covid_abm.Sim()
        sim0.set_seed(5)
        sim0.run(verbose=verbose)
        
        # Not removing people
        sim1 = covid_abm.Sim()
        sim1.set_seed(5)
        sim1['evac_positives'] = 0
        sim1.run(verbose=verbose)
        
        # 100% effective quarantine
        sim2 = covid_abm.Sim()
        sim2.set_seed(5)
        sim2['quarantine_eff'] = 0.0
        sim2.run(verbose=verbose)
        
        # No quarantine
        sim3 = covid_abm.Sim()
        sim3.set_seed(5)
        sim3['quarantine_eff'] = 1.0
        sim3.run(verbose=verbose)
        
        data = sc.odict({'Status quo': sim0, 
                'No disembarkation': sim1, 
                '100% effective quarantine':sim2, 
                'No quarantine':sim3
                })
        
        sc.saveobj('scenario_data.obj', data)
    
    else:
        data = sc.loadobj('scenario_data.obj')
        
    font_size = 18
    fig = pl.figure(figsize=(26,12))
    pl.rcParams['font.size'] = font_size
    plot_args    = {'lw':4, 'alpha':0.7}
    
    for key,datum in data.items():
        if key != 'No quarantine':
            pl.plot(datum.results['t'], datum.results['cum_exposed'], label=key, **plot_args)
            pl.ylabel('Count')
            pl.xlabel('Days since index case')
    
    covid_abm.fixaxis(datum)
    pl.legend()
    pl.savefig(filename)
    
    
    
    