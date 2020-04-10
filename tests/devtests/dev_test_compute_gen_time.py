''' Test/example for calculating the generation time. '''
import sciris as sc
import covasim as cv

pars = {}
pars['n_days'] = 250
pars['pop_infected'] = 1
pars['dur'] = {}
pars['dur']['exp2inf']  = {'dist':'normal_int', 'par1':4, 'par2':0} # Duration from exposed to infectious
pars['dur']['inf2sym']  = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from infectious to symptomatic
pars['dur']['sym2sev']  = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from symptomatic to severe symptoms
pars['dur']['sev2crit'] = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from severe symptoms to requiring ICU

# Duration parameters: time for disease recovery
pars['dur']['asym2rec'] = {'dist':'normal_int', 'par1':0,  'par2':0} # Duration for asymptomatics to recover
pars['dur']['mild2rec'] = {'dist':'normal_int', 'par1':8,  'par2':0} # Duration from mild symptoms to recovered
pars['dur']['sev2rec']  = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from severe symptoms to recovered - leads to mean total disease time of
pars['dur']['crit2rec'] = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from critical symptoms to recovered
pars['dur']['crit2die'] = {'dist':'normal_int', 'par1':0,  'par2':0} # Duration from critical symptoms to death

pars['OR_no_treat']     = 1.0  # Odds ratio for how much more likely people are to die if no treatment available
pars['rel_symp_prob']   = 2.0  # Scale factor for proportion of symptomatic cases
pars['rel_severe_prob'] = 0  # Scale factor for proportion of symptomatic cases that become severe
pars['rel_crit_prob']   = 0  # Scale factor for proportion of severe cases that become critical
pars['rel_death_prob']  = 0  # Scale factor for proportion of critical cases that result in death
pars['prog_by_age']     = False    
    
pars['rand_seed'] = 1

print('Making sim...')
sim1 = cv.Sim(pars=pars)
sim1.run()
sim1.plot()