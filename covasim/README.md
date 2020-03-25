# Base model

The base model includes the core functionality of Covasim.

## Parameters

This section describes the expected behavior of each parameter in the model. Note: the term "overall infection rate" can be explored using `sim.results['doubling_time']`.

### Simulation parameters
* `pars['scale']`: Multiplicative scale for results. Test: run 2 sims, set to 10, `sim.results['cum_exposed']` in 2nd sim should be 10x higher than first.
* `pars['n']`: Nmber of people in the simulation. Test: `len(sim.people)` should equal this number.
* `pars['n_infected']`: Initial number of people infected. Test: if 0, there should be no infections; if equals `n`, should be no _new_ infections.
* `pars['start_day']`: The calendar date of the start day of the simulation.
* `pars['n_days']`: The number of days to simulate. Test: `len(sim.results['t'].values)` should equal this.
* `pars['seed']`: Random seed for the simulation. Test: two simulations with the same seed should produce identical results _except for_ person UIDs; otherwise, different.
* `pars['verbose']`: Level of detail to print (no test).
* `pars['usepopdata']`: Whether or not to use the `synthpops` library for contact matrices. Consult that library's documentation for tests.
* `pars['timelimit']`: Stop simulation if it exceeds this duration. Test: set to a small number (e.g. 1) and choose a large `n`/`n_days`.
* `pars['stop_func']`: User-defined stopping function (no test).

### Disease transmission
* `pars['beta']`: Transmissibility per contact. Test: set to 0 for no infections, set to 1 for ≈`contacts` infections per day (will not be exactly equal due to overlap and other effects)
* `pars['asym_prop']`: Proportion asymptomatic. Test: set to 1 and set `asym_factor` to 0 for no infections.
* `pars['asym_factor']`: Effect of asymptomaticity on transmission. Test: see above.
* `pars['diag_factor']`: Effect of diagnosis on transmission. Highly complex; no unit test for now.
* `pars['cont_factor']`: Effect of being a known contact  on transmission. Highly complex; no unit test for now.
* `pars['contacts']`: Number of contacts per person. Test: set to 0 for no infections; infection rate should scale roughly linearly with this parameter.
* `pars['beta_pop']`: Transmissibility per contact, population-specific. Dependent on `synthpops`. Test: set all to 0 for no infections; infection rate should scale roughly linearly with these parameters.
* `pars['contacts_pop']`: Number of contacts per person, popularion-specific. See `synthpops` documentation for tests.

### Disease progression
* `pars['serial']`: Serial interval (duration between infection and infectiousness). Test: set to `>n` for no transmission. Overall infection rate should scale roughly linearly with this parameter. 
* `pars['serial_std']`: Standard deviation of serial interval. Test: set to 0, set `pars['beta']=1`, and confirm that infections occur `pars['serial']` days apart.
* `pars['incub']`: Incubation period for people who are symptomatic. Highly complex; no unit test for now.
* `pars['incub_std']`: See above.
* `pars['dur']`: Duration of infectiousness. Test: overall infection rate should scale roughly linearly with this parameter. 
* `pars['dur_std']`: Standard deviation of duration of infectiousness. Highly complex; no unit test for now.

### Testing
* `pars['daily_tests']`: Number of daily tests. Tests: set to 0 and diagnoses should be 0; set to `n` and diagnoses should equal number infected if `pars['sensitivity']`=1.
* `pars['sensitivity']`: Sensitivity of the test. Test: set to 0 and diagnoses should be 0.
* `pars['sympt_test']`: Excess probability of testing if symptomatic. Test: for `pars['daily_tests']` ≪ `n`, setting `pars['sympt_test']` ≫ 1 should lead to more diagnoses.
* `pars['trace_test']`: Excess probability of testing if a known contact is infected. Test: for `pars['daily_tests']` ≪ `n`, setting `pars['trace_test']` ≫ 1 should lead to more diagnoses.

### Mortality
* `pars['timetodie']`: Duration of time until death.  Test: set `pars['timetodie_std']=0` and `pars['timetodie']>pars['n_days']`, and there should be no deaths even with `pars['default_cfr']=1`.
* `pars['timetodie_std']`: Standard deviation of death. Tests: set to 0, set `pars['cfr_by_age']=0` and `pars['default_cfr']=1` and `pars['n_infected']=pars['n']`, and everyone should die after this many days. Increase it and it should spread.
* `pars['cfr_by_age']`: Whether or not to use age-dependent CFR. Overrides `pars['default_cfr']`. Test: set to `True`, and make two populations for two simulations, `sim1` with everyone aged 20 and `sim2` with everyone aged 80. Assuming sufficient population size, `sim1.results['cum_deaths'][-1] < sim2.results['cum_deaths'][-1]`.
* `pars['default_cfr']`: Case fatality rate. Test: set `pars['cfr_by_age']` to `False`. Set to 0 and there should be no deaths. Set to 1 and all infected people should die (given enough time).