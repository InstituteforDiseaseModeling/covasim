# Base model

The base model includes the core functionality of Covasim.

## Parameters

This section describes the expected behavior of each parameter in the model.

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
* `pars['beta']`: Transmissibility per contact. Test: set to 0 for no infections, set to 1 for ~`contacts` infections per day (will not be exactly equal due to overlap and other effects)
* `pars['asym_prop']`: Proportion asymptomatic. Test: set to 1 and set `asym_factor` to 0 for no infections.
* `pars['asym_factor']`: Effect of asymptomaticity on transmission. Test: see above.
* `pars['diag_factor']`: Effect of diagnosis on transmission. Highly complex; no unit test for now.
* `pars['cont_factor']`: Effect of being a known contact  on transmission. Highly complex; no unit test for now.
* `pars['contacts']`: Number of contacts per person. Test: set to 0 for no infections; infection rate should scale roughly linearly with this parameter.
* `pars['beta_pop']`: Transmissibility per contact, population-specific. Dependent on `synthpops`. Test: set all to 0 for no infections; infection rate should scale roughly linearly with these parameters
* `pars['contacts_pop']`: Number of contacts per person, popularion-specific. See `synthpops` documentation for tests.

### Disease progression
    pars['serial']         = 4.0 # Serial interval: days after exposure before a person can infect others (see e.g. https://www.ncbi.nlm.nih.gov/pubmed/32145466)
    pars['serial_std']     = 1.0 # Standard deviation of the serial interval
    pars['incub']          = 5.0 # Incubation period: days until an exposed person develops symptoms
    pars['incub_std']      = 1.0 # Standard deviation of the incubation period
    pars['dur']            = 8 # Using Mike's Snohomish number
    pars['dur_std']        = 2 # Variance in duration

### Testing
    pars['daily_tests']    = [0.01*pars['n']]*pars['n_days'] # If there's no testing data, optionally define a list of daily tests here. Remember this gets scaled by pars['scale']. Here we say 1% of the population is tested
    pars['sensitivity']    = 1.0 # Probability of a true positive, estimated
    pars['sympt_test']     = 100.0 # Multiply testing probability by this factor for symptomatic cases
    pars['trace_test']     = 1.0 # Multiply testing probability by this factor for contacts of known positives -- baseline assumes no contact tracing

### Mortality
    pars['timetodie']      = 21 # Days until death
    pars['timetodie_std']  = 2 # STD
    pars['cfr_by_age']     = 0 # Whether or not to use age-specific case fatality
    pars['default_cfr']    = 0.016 # Default overall case fatality rate if not using age-specific values