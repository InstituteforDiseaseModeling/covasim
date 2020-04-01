# Parameters

This file describes the expected behavior of each parameter in the model. Note: the term "overall infection rate" can be explored using `sim.results['doubling_time']` and `sim.results['r_eff']` (a higher infection rate means lower doubling times and higher _R\_eff_), as well as by simply looking at the epidemic curves.

## Simulation parameters
* `scale`      = Multiplicative scale for results. Test: run 2 sims, set to 10, `sim.results['cum_exposed']` in 2nd sim should be 10x higher than first.
* `n`          = Nmber of people in the simulation. Test: `len(sim.people)` should equal this number.
* `n_infected` = Initial number of people infected. Test: if 0, there should be no infections; if equals `n`, should be no _new_ infections.
* `start_day`  = The calendar date of the start day of the simulation.
* `n_days`     = The number of days to simulate. Test: `len(sim.results['t.values'])` should equal this.
* `seed`       = Random seed for the simulation. Test: two simulations with the same seed should produce identical results _except for_ person UIDs; otherwise, different.
* `verbose`    = Level of detail to print (no test).
* `usepopdata` = Whether or not to use the `synthpops` library for contact matrices. Consult that library's documentation for tests.
* `timelimit`  = Stop simulation if it exceeds this duration. Test: set to a small number (e.g. 1) and choose a large `n`/`n_days`.
* `stop_func`  = User-defined stopping function (no test).
* `window`     = Integration window for calculatingthe doubling time; does not affect the simulation otherwise.

## Disease transmission
* `beta`         = Transmissibility per contact. Test: set to 0 for no infections, set to 1 for ≈`contacts` infections per day (will not be exactly equal due to overlap and other effects)
* `asymp_factor` = Effect of asymptomaticity on transmission.
* `diag_factor`  = Effect of diagnosis on transmission.
* `cont_factor`  = Effect of being a known contact  on transmission.
* `contacts`     = Number of contacts per person. Test: set to 0 for no infections. Infection rate should scale roughly linearly with this parameter.
* `beta_pop`     = Transmissibility per contact, population-specific. Dependent on `synthpops`. Test: set all to 0 for no infections. Infection rate should scale roughly linearly with these parameters.
* `contacts_pop` = Number of contacts per person, population-specific. See `synthpops` documentation for tests.

## Duration parameters
* `exp2inf`  = Duration from exposed to infectious
* `inf2sym`  = Duration from infectious to symptomatic
* `sym2sev`  = Duration from symptomatic to severe symptoms
* `sev2crit` = Duration from severe symptoms to requiring ICU
* `asym2rec` = Duration for asymptomatics to recover
* `mild2rec` = Duration from mild symptoms to recovered
* `sev2rec`  = Duration from severe symptoms to recovered
* `crit2rec` = Duration from critical symptoms to recovered
* `crit2die` = Duration from critical symptoms to death

## Severity parameters: probabilities of symptom progression
* `prog_by_age`     = Whether or not to use age-specific probabilities of prognosis (symptoms/severe symptoms/death)
* `rel_symp_prob`   = If not using age-specific values: relative proportion of symptomatic cases
* `rel_severe_prob` = If not using age-specific values: relative proportion of symptomatic cases that become severe
* `rel_crit_prob`   = If not using age-specific values: relative proportion of severe cases that become critical
* `rel_death_prob`  = If not using age-specific values: relative proportion of critical cases that result in death
* `OR_no_treat`     = Odds ratio for how much more likely people are to die if no treatment available

## Events and interventions
* `interventions` = List of Intervention instances
* `interv_func`   = Custom intervention function

## Health system parameters
* `n_beds` = Baseline assumption is that there's enough beds for the whole population (i.e., no constraints)


## Events and interventions
* `interventions`: A list of `Intervention` objects; see `examples/run_scenarios.py` for example usage.
* `interv_func`: A custom intervention function; see `tests/dev_test_synthpops.py` for an example.
