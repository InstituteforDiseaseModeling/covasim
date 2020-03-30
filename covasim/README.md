# Parameters

This file describes the expected behavior of each parameter in the model. Note: the term "overall infection rate" can be explored using `sim.results['doubling_time']` and `sim.results['r_eff']` (a higher infection rate means lower doubling times and higher _R\_eff_), as well as by simply looking at the epidemic curves.

## Simulation parameters
* `scale`: Multiplicative scale for results. Test: run 2 sims, set to 10, `sim.results['cum_exposed']` in 2nd sim should be 10x higher than first.
* `n`: Nmber of people in the simulation. Test: `len(sim.people)` should equal this number.
* `n_infected`: Initial number of people infected. Test: if 0, there should be no infections; if equals `n`, should be no _new_ infections.
* `start_day`: The calendar date of the start day of the simulation.
* `n_days`: The number of days to simulate. Test: `len(sim.results['t.values'])` should equal this.
* `seed`: Random seed for the simulation. Test: two simulations with the same seed should produce identical results _except for_ person UIDs; otherwise, different.
* `verbose`: Level of detail to print (no test).
* `usepopdata`: Whether or not to use the `synthpops` library for contact matrices. Consult that library's documentation for tests.
* `timelimit`: Stop simulation if it exceeds this duration. Test: set to a small number (e.g. 1) and choose a large `n`/`n_days`.
* `stop_func`: User-defined stopping function (no test).
* `window`: Integration window for calculatingthe doubling time; does not affect the simulation otherwise.

## Disease transmission
* `beta`: Transmissibility per contact. Test: set to 0 for no infections, set to 1 for ≈`contacts` infections per day (will not be exactly equal due to overlap and other effects)
* `asym_factor`: Effect of asymptomaticity on transmission.
* `diag_factor`: Effect of diagnosis on transmission.
* `cont_factor`: Effect of being a known contact  on transmission.
* `contacts`: Number of contacts per person. Test: set to 0 for no infections. Infection rate should scale roughly linearly with this parameter.
* `beta_pop`: Transmissibility per contact, population-specific. Dependent on `synthpops`. Test: set all to 0 for no infections. Infection rate should scale roughly linearly with these parameters.
* `contacts_pop`: Number of contacts per person, population-specific. See `synthpops` documentation for tests.

## Disease progression
* `serial`: Serial interval (duration between infection and infectiousness). Test: set to `>n` for no transmission. Overall infection rate should scale roughly linearly with this parameter.
* `serial_std`: Standard deviation of serial interval. Test: set to 0, set `beta=1`, and confirm that infections occur `serial` days apart.
* `incub`: Incubation period for people who are symptomatic. Highly complex; no unit test for now.
* `incub_std`: See above.
* `dur`: Duration of infectiousness. Test: overall infection rate should scale roughly linearly with this parameter.
* `dur_std`: Standard deviation of duration of infectiousness. Highly complex; no unit test for now.

## Mortality and severity
* `timetodie`: Duration of time until death.  Test: set `timetodie_std=0` and `timetodie>n_days`, and there should be no deaths even with `default_cfr=1`.
* `timetodie_std`: Standard deviation of death. Tests: set to 0, set `prog_by_age=0` and `default_death_prob=1` and `n_infected=n`, and everyone should die after this many days. Increase it and it should spread.
* `prog_by_age`: Whether or not to use age-dependent CFR. Overrides `default_death_prob`. Test: set to `True`, and make two populations for two simulations, `sim1` with everyone aged 20 and `sim2` with everyone aged 80. Assuming sufficient population size, `sim1.results['cum_deaths'][-1] < sim2.results['cum_deaths'][-1]`.
* `default_symp_prob`: Probability of developing symptoms; see test below.
* `default_severe_prob`: Probability of developing a severe case; see test below.
* `default_death_prob`: Case fatality rate. Test: set `prog_by_age` to `False`. Set to 0 and there should be no deaths. Set to 1 along with `default_symp_prob` and `default_severe_prob`, and all infected people should die (given enough time).

## Events and interventions
* `interventions`: A list of `Intervention` objects; see `examples/run_scenarios.py` for example usage.
* `interv_func`: A custom intervention function; see `tests/dev_test_synthpops.py` for an example.
