import sciris as sc
import covasim as cv
import numpy as np

runs = 301
r0_const = np.zeros(runs)
r0_twolevel = np.zeros(runs)
base_pars = sc.objdict(
        n_days       = 12,   # Number of days to simulate
        asymp_factor = 1, # Multiply beta by this factor for asymptomatic cases
        diag_factor  = 1, # Multiply beta by this factor for diganosed cases -- baseline assumes complete isolation
        cont_factor  = 1.0,
        verbose = 0,
        pop_infected = 10
)
base_pars['dur'] = {}
base_pars['dur']['exp2inf']  = {'dist':'normal_int', 'par1':4, 'par2':0} # Duration from exposed to infectious
base_pars['dur']['inf2sym']  = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from infectious to symptomatic
base_pars['dur']['sym2sev']  = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from symptomatic to severe symptoms
base_pars['dur']['sev2crit'] = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from severe symptoms to requiring ICU

# Duration parameters: time for disease recovery
base_pars['dur']['asym2rec'] = {'dist':'normal_int', 'par1':0,  'par2':0} # Duration for asymptomatics to recover
base_pars['dur']['mild2rec'] = {'dist':'normal_int', 'par1':8,  'par2':0} # Duration from mild symptoms to recovered
base_pars['dur']['sev2rec']  = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from severe symptoms to recovered - leads to mean total disease time of
base_pars['dur']['crit2rec'] = {'dist':'normal_int', 'par1':0, 'par2':0} # Duration from critical symptoms to recovered
base_pars['dur']['crit2die'] = {'dist':'normal_int', 'par1':0,  'par2':0} # Duration from critical symptoms to death

base_pars['OR_no_treat']     = 1.0  # Odds ratio for how much more likely people are to die if no treatment available
base_pars['rel_symp_prob']   = 1.0  # Scale factor for proportion of symptomatic cases
base_pars['rel_severe_prob'] = 0  # Scale factor for proportion of symptomatic cases that become severe
base_pars['rel_crit_prob']   = 0  # Scale factor for proportion of severe cases that become critical
base_pars['rel_death_prob']  = 0  # Scale factor for proportion of critical cases that result in death
base_pars['prog_by_age']     = False    
    
for i in range(runs):
    # Configure the sim -- can also just use a normal dictionary
    pars = base_pars
    pars['rand_seed'] = i
    print('Making sim ', i, '...')
    sim1 = cv.Sim(pars=pars)
    sim1.run()
    r0_const[i] = len(sim1.people[0].infected)
    pars = base_pars
    pars['rand_seed'] = i
    pars['viral_distro'] = {'dist':'twolevel', 'frac':.5, 'ratio':2}
    sim2 = cv.Sim(pars=pars)
    sim2.run()
    r0_twolevel[i] = len(sim2.people[0].infected)

print('R0 constant viral load: ', np.mean(r0_const), ' +- ', np.std(r0_const))
print('R0 two level viral load: ', np.mean(r0_twolevel), ' +- ', np.std(r0_twolevel))

import matplotlib.pyplot as plt
hist1 = plt.hist(r0_const, bins=np.arange(-0.5, 11.5))
hist2 = plt.hist(r0_twolevel, bins=np.arange(-0.5, 11.5))
plt.show()
print(abs(hist1[0]-hist2[0])/hist1[0])
