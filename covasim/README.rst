==========
Parameters
==========

This file describes the expected behavior of each parameter in the model. Note: the term "overall infection rate" can be explored using ``sim.results['doubling_time']`` and ``sim.results['r_eff']`` (a higher infection rate means lower doubling times and higher *R\_eff*), as well as by simply looking at the epidemic curves.

Population parameters
---------------------
* ``pop_size``     = Number ultimately susceptible to CoV
* ``pop_infected`` = Number of initial infections
* ``pop_type``     = What type of population data to use -- random (fastest), synthpops (best), hybrid (compromise), or clustered (not recommended)
* ``location``     = What location to load data from -- default Seattle

Simulation parameters
---------------------
* ``start_day``  = Start day of the simulation
* ``end_day``    = End day of the simulation
* ``n_days``     = Number of days to run, if end_day isn't specified
* ``rand_seed``  = Random seed, if None, don't reset
* ``verbose``    = Whether or not to display information during the run -- options are 0 (silent), 1 (default), 2 (everything)

Rescaling parameters
--------------------
* ``pop_scale``         = Factor by which to scale the population -- e.g. 1000 with pop_size = 10e3 means a population of 10m
* ``rescale``           = Enable dynamic rescaling of the population
* ``rescale_threshold`` = Fraction susceptible population that will trigger rescaling if rescaling
* ``rescale_factor``    = Factor by which we rescale the population

Basic disease transmission
--------------------------
* ``beta``        = Beta per symptomatic contact; absolute
* ``contacts``    = The number of contacts per layer; set below
* ``dynam_layer`` = Which layers are dynamic; set below
* ``beta_layer``  = Transmissibility per layer; set below
* ``n_imports``   = Average daily number of imported cases (actual number is drawn from Poisson distribution)
* ``beta_dist``   = Distribution to draw individual level transmissibility
* ``viral_dist``  = The time varying viral load (transmissibility)

Efficacy of protection measures
-------------------------------
* ``asymp_factor`` = Multiply beta by this factor for asymptomatic cases
* ``diag_factor``  = Multiply beta by this factor for diganosed cases
* ``quar_eff``     = Quarantine multiplier on transmissibility and susceptibility; set below
* ``quar_period``  = Number of days to quarantine for

Time for disease progression
----------------------------
* ``exp2inf``  = Duration from exposed to infectious
* ``inf2sym``  = Duration from infectious to symptomatic
* ``sym2sev``  = Duration from symptomatic to severe symptoms
* ``sev2crit`` = Duration from severe symptoms to requiring ICU

Time for disease recovery
-------------------------
* ``asym2rec`` = Duration for asymptomatics to recover
* ``mild2rec`` = Duration from mild symptoms to recovered
* ``sev2rec``  = Duration from severe symptoms to recovered
* ``crit2rec`` = Duration from critical symptoms to recovered
* ``crit2die`` = Duration from critical symptoms to death

Severity parameters
-------------------
* ``OR_no_treat``     = Odds ratio for how much more likely people are to die if no treatment available
* ``rel_symp_prob``   = Scale factor for proportion of symptomatic cases
* ``rel_severe_prob`` = Scale factor for proportion of symptomatic cases that become severe
* ``rel_crit_prob``   = Scale factor for proportion of severe cases that become critical
* ``rel_death_prob``  = Scale factor for proportion of critical cases that result in death
* ``prog_by_age``     = Whether to set disease progression based on the person's age
* ``prognoses``       = Populate this later

Events and interventions
------------------------
* ``interventions`` = List of Intervention instances
* ``interv_func``   = Custom intervention function
* ``timelimit``     = Time limit for a simulation (seconds)
* ``stopping_func`` = A function to call to stop the sim partway through

Health system parameters
--------------------------
* ``n_beds`` = Baseline assumption is that there's no upper limit on the number of beds i.e. there's enough for everyone