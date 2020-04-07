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
* `age_cutoffs`  = Age bins for severity parameter values
* `symp_probs`   = Overall probability of developing symptoms
* `severe_probs` = Overall probability of developing severe symptoms
* `crit_probs`   = Overall probability of developing critical symptoms
* `death_probs`  = Overall probability of dying
* `OR_no_treat`  = Odds ratio for how much more likely people are to die if no treatment available

## Events and interventions
* `interventions` = List of Intervention instances
* `interv_func`   = Custom intervention function

## Health system parameters
* `n_beds` = Baseline assumption is that there's enough beds for the whole population (i.e., no constraints)


## Events and interventions
* `interventions`: A list of `Intervention` objects; see `examples/run_scenarios.py` for example usage.
* `interv_func`: A custom intervention function; see `tests/dev_test_synthpops.py` for an example.
