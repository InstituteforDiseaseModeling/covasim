==========
What's new
==========

All notable changes to the codebase are documented in this file. As of version 1.3, changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".

.. contents:: **Contents**
   :local:
   :depth: 1


~~~~~~~~~~~~~~~
Latest versions
~~~~~~~~~~~~~~~


Version 1.4.1 (2020-05-29)
--------------------------
- Added ``sim.people.plot()``, which shows the age distribution, and distribution of contacts by age and layer.
- Added ``sim.make_age_histogram()``, as well as the ability to call ``cv.age_histogram(sim)``, as an alternative to adding these as analyzers to a sim.
- Updated ``cv.make_synthpop()`` to pass a random seed to SynthPops (note: requires SynthPops version 0.7.1 or later).
- ``cv.set_seed()`` now also resets ``random.seed()``, to ensure reproducibility among functions that use this (e.g., NetworkX).
- Corrected ``sim.run()`` so ``sim.t`` is left at the last timestep (instead of one more).


Version 1.4.0 (2020-05-28)
--------------------------

This version contains a large number of changes, including two new classes, ``Analyzer`` and ``Fit``, for performing simulation analyses and fitting the model to data, respectively. These changes are described below.

Analysis
^^^^^^^^
- Added a new class, ``Analyzer``, to perform analyses on a simulation.
- Added a new parameter, ``sim['analyzers']``, that operates like ``interventions``: it accepts a list of functions or ``Analyzer`` objects.
- Added two analyzers: ``cv.age_hist`` records age histograms of infections, diagnoses, and deaths; ``cv.snapshot`` makes copies of the ``People`` object at specified points in time.


Fitting
^^^^^^^
- Added a new class, ``cv.Fit()``, that stores information about the fit between the model and the data. "Likelihood" is no longer automatically calculated, but instead "mismatch" can be calculated via ``fit = sim.compute_fit()``.
- The Poisson test that was previously used for the "likelihood" calculation has been deprecated; the new default mismatch is based on normalized absolute error.
- For a plot of how the mismatch is being calculated, use ``fit.plot()``.

MultiSims
^^^^^^^^^
- Added ``multisim.init_sims()``, which is not usually necessary, but can be helpful if you want to create the ``Sim`` objects without running them straight away.
- Added ``multisim.split()``, easily allowing a merged multisim to be split back into its constituent parts (non-merged multisims can also be split). This can be used for example to create several multisims, merge them together, run them all at the same time in parallel, and then split the back for analysis.

Display functions
^^^^^^^^^^^^^^^^^
- Added ``sim.summarize()``, which shows a short review of key sim results (cumulative counts).
- Added ``sim.brief()``, which shows a one-line summary of the sim.
- Added ``multisim.summarize()``, which prints a brief summary of all the constituent sims.

Parameter changes
^^^^^^^^^^^^^^^^^
- Removed the parameter ``interv_func``; instead, intervention functions can now be appended to ``sim['interventions']``.
- Changed the default for the ``rescale`` parameter from ``False`` to ``True``. To return to previous behavior, define ``sim['rescale'] = False`` explicitly.

Other changes
^^^^^^^^^^^^^
- Added ``cv.day()`` convenience function to convert a date to an integer number of days (similar to ``cv.daydiff()``); also modified ``cv.date()`` to be able to handle input more flexibly. While ``sim.day()`` and ``sim.date()`` are still the recommended functions, the same functionality is now also available without a ``Sim`` object available.
- Allowed `cv.load_data()`` to accept non-time-series inputs.
- Added cumulative diagnoses to default plots.
- Moved ``sweeps`` (Weights & Biases) to ``examples/wandb``.
- Refactored cruise ship example to work again.
- Various bugfixes (e.g. to plotting arguments, data scrapers, etc.).
- *Regression information*: To migrate an old parameter set ``pars`` to this version and to restore previoius behavior, use::

    pars['analyzers'] = None # Add the new parameter key
    interv_func = pars.pop('interv_func', None) # Remove the deprecated key
    if interv_func:
        pars['interventions'] = interv_func # If no interventions
        pars['interventions'].append(interv_func) # If other interventions are present
    pars['rescale'] = pars.pop('rescale', False) # Change default to False

- *GitHub info*: PR `569 <https://github.com/amath-idm/covasim/pull/569>`__, previous head ``8b157a2``


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.3.x (1.3.0 – 1.3.5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.3.5 (2020-05-28)
--------------------------
- Added ``swab_delay`` argument to ``cv.test_num()``, allowing a distribution of times between when a person develops symptoms and when they go to be tested (i.e., receive a swab) to be specified.



Version 1.3.4 (2020-05-26)
--------------------------
- Allowed data to be loaded from a dataframe instead of from file.
- Fixed data scrapers to use correct column labels.


Version 1.3.3 (2020-05-26)
--------------------------
- Fixed issue with a loaded population being reloaded when a simulation is re-initialized.
- Fixed issue with the argument ``dateformat`` not being passed to the right plotting routine.
- Fixed issue with MultiSim plotting appearing in separate panels when run in a Jupyter notebook.
- Fixed issue with ``cv.git_info()`` failing to write to file when the calling function could not be found.


Version 1.3.2 (2020-05-25)
--------------------------
- ``People`` and ``popdict`` objects can now be supplied directly to the sim instead of a file name.
- ``git_info()`` and ``check_save_info()`` now include information from the calling script (not just Covasim). They also now include a ``comments`` field to optionally store additional information.


Version 1.3.1 (2020-05-25)
--------------------------
- Modified calculation of ``R_eff`` to include a longer integration period at the beginning, and restored previous method of creating seed infections. 
- Updated default plots to include number of active infections, and removed recoveries.


Version 1.3.0 (2020-05-24)
--------------------------
- Changed the default number of work contacts in hybrid from 8 to 16, and halved beta from 1.4 to 0.7, to better capture superspreading events. *Regression information*: To restore previous behavior, set ``sim['beta_layer']['w'] = 0.14`` and ``sim['contacts']['w'] = 8``.
- Initial infections now occur at a distribution of dates instead of all at once; this fixes the artificial spike in ``R_eff`` that occurred at the very beginning of a simulation. *Regression information*: This change affects results, but was reverted in the next version (1.3.1).
- Changed the definition of age bins in prognoses to be lower limits rather than upper limits. Added an extra set of age bins for 90+.
- Changed population loading and saving to be based on People objects, not popdicts (syntax is exactly the same, although it is recommended to use ``.ppl`` instead of ``.pop`` for these files).
- Added additional random seed resets to population initialization and just before the run so that populations loaded from disk produce identical results to newly created ones. *Regression information*: This affects results by changing the random number stream. In most cases, previous behavior can typically be restored by setting ``sim.run(reset_seed=False)``.
- Added a new convenience method, ``cv.check_save_info()``, which can be put at the top of a script to check the Covasim version and automatically save the Git info to file.
- Added additional methods to ``People`` to retrieve different types of keys: e.g., ``sim.people.state_keys()`` returns all the different states a person can be in (e.g., ``symptomatic``).
- *GitHub info*: PR `557 <https://github.com/amath-idm/covasim/pull/557>`__, previous head ``aac9f1d``


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.2.x (1.2.0 – 1.2.3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.2.3 (2020-05-23)
--------------------------
- Added ``cv.savefig()``, which is an alias to Matplotlib's ``savefig()`` function, but which saves additional metadata in the figure file. This metadata can be loaded with the new ``cv.get_png_metdata()`` function.
- Major changes to ``MultiSim`` plotting, incorporating all the flexibility of both simulation and scenario plotting. By default, with a small number of runs (<= 5), it defaults to scenario-style plotting; else, it defaults to simulation-style plotting.
- Default scenario plotting options were updated (e.g., showing deaths instead of hospitalizations).
- You may merge multiple multisims more merrily now, with e.g. ``msim = cv.MultiSim.merge(msim1, msim2)``.
- Test scripts (e.g. ``tests/run_tests``) have been updated to use ``pytest-parallel``, reducing wall-clock time by a factor of 5.
- *GitHub info*: PR `552 <https://github.com/amath-idm/covasim/pull/552>`__, previous head ``28bf02b``


Version 1.2.2 (2020-05-22)
--------------------------
- Changed the syntax of ``cv.clip_edges()`` to match ``cv.change_beta()``. The old format of intervention ``cv.clip_edges(start_day=d1, end_day=d2, change=c)`` should now be written as ``cv.clip_edges(days=[d1, d2], changes=[c, 1.0])``.
- Changed the syntax for the transmission tree: it now takes the ``Sim`` object rather than the ``People`` object, and typical usage is now ``tt = sim.make_transtree()``.
- Plots now default to a maximum of 4 rows; this can be overridden using the ``n_cols`` argument, e.g. ``sim.plot(to_plot='overview', n_cols=2)``.
- Various bugs with ``MultiSim`` plotting were fixed.
- *GitHub info*: PR `551 <https://github.com/amath-idm/covasim/pull/551>`__, previous head ``07009eb``


Version 1.2.1 (2020-05-21)
--------------------------
- Added influenza-like illness (ILI) symptoms to testing interventions. If nonzero, this reduces the effectiveness of symptomatic testing, because you cannot distinguish between people who are symptomatic with COVID and people with other ILI symptoms.
- Removed an unneeded ``copy()`` in ``single_run()`` because multiprocessing always produces copies of objects via the pickling process.
- *GitHub info*: PR `541 <https://github.com/amath-idm/covasim/pull/541>`__, previous head ``9b2dbfb``


Version 1.2.0 (2020-05-20)
--------------------------
- Since parameters can be modified during the run, previously, the sim could not be rerun with the guarantee that the results would be the same. ``sim.run()`` now has a ``restore_pars`` argument (default true), which makes a copy of the parameters just prior to the run to ensure reproducibility.
- In plotting, by default, data points are now slightly transparent and behind the lines to improve visibility of the model curve.
- Interventions now have a ``label`` attribute, which can be helpful for finding them if many are used, e.g. ``[interv if interv.label=='Close schools' for interv in sim['interventions']``. There is also a new method, ``intervention.disp()``, which prints out detailed information about an intervention object.
- Subtargeting of particular people in testing interventions can now be done via a function that gets called dynamically, avoiding the need to initialize the sim prior to creating the intervention.
- Layer keys are now stored inside the ``popdict``, for greater consistency handling loaded populations. Layer key handling has been simplified and made more robust.
- Loading and saving a population is now controlled by the ``Sim`` object, not by the ``sim.initialize()`` method. Instead of ``sim = cv.Sim(); sim.initialize(save_pop=True)``, you can now simply do ``sim = cv.Sim(save_pop=True``, and it will save when the sim is initialized.
- Added prevalence and incidence as results.
- Added ``sim.scaled_pop_size``, which is the population size (the number of agents) times the population scale factor. This corresponds to the "actual" population size being modeled.
- Removed the numerical artifact at the beginning and end of the ``R_eff`` calculation due to the smoothing kernel, and confirmed that the spike in ``R_eff`` often seen at the beginning is due to the way the seed infectious progress from exposed to infectious, and not from a bug.
- Added more flexibility to plotting, including a new ``show_args`` keyword, allowing particular aspects of plotting (e.g., the data or interventions) to be turned on or off.
- Moved the cruise ship code from the core folder into the examples folder.
- *GitHub info*: PR `538 <https://github.com/amath-idm/covasim/pull/538>`__, previous head ``451f410``


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.1.x (1.1.0 – 1.1.7)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.1.7 (2020-05-19)
--------------------------
- Diagnoses are now reported on the day the test was conducted, not the day the person gets their diagnosis. This is to better align with data (which is reported this way), and to avoid a bug in which test yield could be >100%. A new attribute, ``date_pos_test``, was added to the ``sim.people`` object in order to track the date on which a person is given the test which will (after ``test_delay`` days) come back positive.
- An "overview" plotting feature has been added for sims and scenarios: simply use ``sim.plot(to_plot='overview')`` to use. This plots almost all of the simulation outputs on one screen.
- It is now possible to set ``pop_type = None`` if you are supplying a custom population.
- Population creation functions (including the ``People`` class) have been tidied up with additional docstrings added.
- Duplication between pre- and post-step state checking has been removed.
- *GitHub info*: PR `537 <https://github.com/amath-idm/covasim/pull/537>`__, previous head ``2d55c38``



Version 1.1.6 (2020-05-19)
--------------------------
- Created an ``analysis.py`` file to support different types of analysis.
- Moved ``transtree`` from ``sim.people`` into its own class: thus instead of ``sim.people.make_detailed_transtree()``, the new syntax is ``tt = cv.TransTree(sim.people)``.
- *GitHub info*: PR `531 <https://github.com/amath-idm/covasim/pull/531>`__, previous head ``998116c``


Version 1.1.5 (2020-05-18)
--------------------------
- Added extra flexibility for targeting interventions by index of a person, for example, by age.


Version 1.1.4 (2020-05-18)
--------------------------
- Added a new hospital bed capacity constraint and renamed health system capacity parameters. To migrate an older set of parameters to this version, set::

    pars['no_icu_factor']  = pars.pop('OR_no_treat')
    pars['n_beds_icu']     = pars.pop('n_beds')
    pars['no_hosp_factor'] = 1.0
    pars['n_beds_hosp']    = None

- Removed the ``bed_capacity`` result.
- *GitHub info*: PR `510 <https://github.com/amath-idm/covasim/pull/510>`__, previous head ``0f6d48c``


Version 1.1.3 (2020-05-18)
--------------------------
- Improved the how "layer parameters" (e.g., ``beta_layer``) are initialized.
- Allowed arbitrary arguments to be passed to SynthPops via ``cv.make_synthpop``.


Version 1.1.2 (2020-05-18)
--------------------------
- Added a new result, ``test_yield``, which is the number of diagnoses divided by the number of cases each day.
- Minor improvements to date handling and plotting.


Version 1.1.1 (2020-05-13)
--------------------------
- Refactored the contact tracing and quarantining functions, to fixed a bug (introduced in v1.1.0) in which some people who went into quarantine never came out of quarantine.
- Changed initialization so seed infections are now sampled randomly from the population, rather than the first ``pop_infected`` agents. Since ``hybrid`` also uses consecutive indices for constructing households, this was causing some households to be fully infected on initialization, while all other households had no infections.
- Updated the default ``rescale_factor`` from 2.0 to 1.2, since large amounts of rescaling cause noticeable "blips" in inhomogeneous networks (e.g., a population where some households are 100% infected and most are 0% infected).
- Added ability to pass plotting arguments to ``intervention.plot()``.
- Removed default noise in scenarios (restore previous behavior by setting ``metapars = dict(noise=0.1)``).
- Refactored and renamed computed results (e.g., summary stats) in the Sim class.
- *GitHub info*: PR `513 <https://github.com/amath-idm/covasim/pull/513>`__, previous head ``973801a``


Version 1.1.0 (2020-05-12)
--------------------------
- Renamed the parameter ``diag_factor`` to ``iso_factor``, and converted it to a dictionary by layer.
- Renamed the parameter ``quar_eff`` to ``quar_factor`` (but otherwise left it unchanged).
- Added the option for presumptive isolation and quarantine in testing interventions.
- Fixed a bug whereby people who had been in quarantine and were then diagnosed had both diagnosis and quarantine factors applied.
- *GitHub info*: PR `502 <https://github.com/amath-idm/covasim/pull/502>`__, previous head ``0230383``


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.0.x (1.0.0 – 1.0.3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.0.3 (2020-05-11)
--------------------------
- Added an extra output of ``make_microstructured_contacts()`` to store each person's cluster identifier. Currently, this is only supported for the ``hybrid`` population type, but in future versions, ``synthpops`` will also be supported.
- Removed the ``directed`` argument from population creation functions since it is no longer supported in the model.


Version 1.0.2 (2020-05-10)
--------------------------
- Added uncertainty to the ``plot_result()`` method of MultiSims.
- Added documentation and webapp links to the paper.


Version 1.0.1 (2020-05-09)
--------------------------
- Added argument ``as_date`` for ``sim.date()`` to return a ``datetime`` object instead of a string.
- Fixed plotting of interventions in the webapp.
- Removed default 1-hour time limit for simulations.
- *GitHub info*: PR `490 <https://github.com/amath-idm/covasim/pull/490>`__, previous head ``9a6c23b``


Version 1.0.0 (2020-05-08)
--------------------------
- Official release of Covasim.
- Made scenario and simulation plotting more flexible: ``to_plot`` can now simply be a list of results keys, e.g. ``cum_deaths``.
- Added additional tests, increasing test coverage from 67% to 92%.
- Fixed bug in ``cv.save()``.
- Added ``reset()`` to MultiSim that undoes a ``reduce()`` or ``combine()`` call.
- General code cleaning: made exceptions raised more consistent, removed unused functions, etc.
- *GitHub info*: PR `487 <https://github.com/amath-idm/covasim/pull/487>`__, previous head ``c8ca32d``


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prerelease versions (0.27.0 – 0.32.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 0.32.1 (2020-05-06)
---------------------------
- Allow ``until`` to be a date, e.g. ``sim.run(until='2020-05-06')``.
- Added ``ipywidgets`` dependency since otherwise the webapp breaks due to a `bug <https://github.com/plotly/plotly.py/issues/2443>`__ with the latest Plotly version (4.7).


Version 0.32.0 (2020-05-05)
---------------------------
- Changed the edges of the contact network from being directed to undirected, halving the amount of memory required and making contact tracing and edge clipping more realistic.
- Added comorbidities to the prognoses parameters.
- *GitHub info*: PR `482 <https://github.com/amath-idm/covasim/pull/482>`__, previous head ``029585f``


Version 0.31.0 (2020-05-05)
---------------------------
- Added age-susceptible odds ratios, and modified severe and critical progression probabilities. To compensate, default ``beta`` has been increased from 0.015 to 0.016. To restore previous behavior (which was based on the `Imperial paper <https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf>`__), set ``beta=0.015`` and set the following values in ``sim.pars['prognoses']``::

    sus_ORs[:]   = 1.0
    severe_probs = np.array([0.00100, 0.00100, 0.01100, 0.03400, 0.04300, 0.08200, 0.11800, 0.16600, 0.18400])
    crit_probs   = np.array([0.00004, 0.00011, 0.00050, 0.00123, 0.00214, 0.00800, 0.02750, 0.06000, 0.10333])

- Relative susceptibility and transmissibility (i.e., ``sim.people.rel_sus``) are now set when the population is initialized (before, they were modified dynamically when a person became infected or recovered). This means that modifying them before a simulation starts, or during a simulation, should be more robust.
- Reordered results dictionary to start with cumulative counts.
- ``sim.export_pars()`` now accepts a filename to save to.
- Added a ``tests/regression`` folder with previous versions of default parameter values.
- Changed ``pars['n_beds']`` to interpret 0 or ``None`` as no bed constraint.
- *GitHub info*: PR `480 <https://github.com/amath-idm/covasim/pull/480>`__, previous head ``c7171f8``


Version 0.30.4 (2020-05-04)
---------------------------
- Changed the detailed transmission tree (``sim.people.transtree.detailed``) to include much more information.
- Added animation method to transmission tree: ``sim.people.transtree.animate()``.
- Added support to generate populations on the fly in SynthPops.
- Adjusted the default arguments for ``test_prob`` and fixed a bug with ``test_num`` not accepting date input.
- Added ``tests/devtests/intervention_showcase.py``, using and comparing all available interventions.


Version 0.30.3 (2020-05-03)
---------------------------
- Fixed bugs in dynamic scaling; see ``tests/devtests/dev_test_rescaling.py``. When using ``pop_scale>1``, the recommendation is now to use ``rescale=True``.
- In ``cv.test_num()``, renamed argument from ``sympt_test`` to ``symp_test`` for consistency.
- Added ``plot_compare()`` method to ``MultiSim``.
- Added ``labels`` arguments to plotting methods, to allow custom labels to be used.


Version 0.30.2 (2020-05-02)
---------------------------
- Updated ``r_eff`` to use a new method based on daily new infections. The previous version, where infections were counted from when someone recovered or died, is available as ``sim.compute_r_eff(method='outcome')``, while the traditional method, where infections are counted from the day someone becomes infectious, is available via ``sim.compute_r_eff(method='infectious')``.


Version 0.30.1 (2020-05-02)
---------------------------
- Added ``end_day`` as a parameter, allowing an end date to be specified instead of a number of days.
- ``Sim.run()`` now displays the date being simulated.
- Added a ``par_args`` arugument to ``multi_run()``, allowing arguments (e.g. ``ncpus``) to be passed to ``sc.parallelize()``.
- Added a ``compare()`` method to multisims and stopped people from being saved by default.
- Fixed bug whereby intervention were not getting initialized if they were added to a sim after it was initialized.


Version 0.30.0 (2020-05-02)
---------------------------
- Added new ``MultiSim`` class for plotting a single simulation with uncertainty.
- Added ``low`` and ``high`` attributes to the ``Result`` object.
- Refactored plotting to increase consistency between ``sim.plot()``, ``sim.plot_result()``, ``scens.plot()``, and ``multisim.plot()``.
- Doubling time calculation defaults have been updated to use a window of 3 days and a maximum of 30 days.
- Added an ``until`` argument to ``sim.run()``, to make it easier to run a partially completed sim and then resume. See ``tests/devtests/test_run_until.py``.
- Fixed a bug whereby ``cv.clip_edges()`` with no end day specified resulted in large sim files when saved.


Version 0.29.9 (2020-04-28)
---------------------------
- Fixed bug in which people who had been tested and since recovered were not being diagnosed.
- Updated definition of "Time to die" parameter in the webapp.


Version 0.29.8 (2020-04-28)
---------------------------
- Updated webapp UI with more detail on and control over interventions.


Version 0.29.7 (2020-04-27)
---------------------------
- New functions ``cv.date()`` and ``cv.daydiff()`` have been added, to ease handling of dates of different formats.
- Defaults are now functions rather than dictionaries, specifically: ``cv.default_sim_plots`` is now ``cv.get_sim_plots()``; ``cv.default_scen_plots`` is now ``cv.get_scen_plots()``; and ``cv.default_colors`` is now ``cv.get_colors()``.
- Interventions now have a ``do_plot`` kwarg, which if ``False`` will disable their plotting.
- The example scenario (``examples/run_scenario.py``) has been rewritten to include a test-trace-quarantine example.


Version 0.29.6 (2020-04-27)
---------------------------
- Updated to use Sciris v0.17.0, to fix JSON export issues and improve ``KeyError`` messages.


Version 0.29.5 (2020-04-26)
---------------------------
- Fixed bug whereby layer betas were applied twice, and updated default values.
- Includes individual-level viral load (to use previous results, set ``pars['beta_dist'] = {'dist':'lognormal','par1':1.0, 'par2':0.0}`` and ``pars['viral_dist']  = {'frac_time':0.0, 'load_ratio':1, 'high_cap':0}``).
- Updated parameter values (mostly durations) based on revised literature review.
- Added ``sim.export_pars()`` and ``sim.export_results()`` methods.
- Interventions can now be converted to/from JSON -- automatically when loading a parameters dictionary into a sim, or manually using ``cv.InterventionDict()``.
- Improvements to transmission trees: can now make a detailed tree with ``sim.people.make_detailed_transtree()`` (replacing ``sim.people.transtree.make_detailed(sim.people)``), and can plot via ``sim.people.transtree.plot()``.
- Improved date handling, so most functions are now agnostic as to whether a date string, datetime object, or number of days is provided; new functions: ``sim.day()`` converts dates to days, ``sim.date()`` (formerly ``sim.inds2dates()``) converts days to dates, and ``sim.daydiff()`` computes the number of days between two dates.


Version 0.28.8 (2020-04-24)
---------------------------
- Includes data on household sizes from various countries.
- Includes age data on US states.
- Changes to interventions to include end as well as start days, and plotting as a default option.
- Adds version checks to loading and introduces a new function ``cv.load()`` to replace e.g. ``cv.Sim.load()``.
- Major layout and functionality changes to the webapp, including country selection (disabled by default).
- Provided access to Plotly graphs via the backend.
- Moved relative probabilities (e.g. ``rel_death_prob``) from population creation to loop so can be modified dynamically.
- Introduced ``cv.clip_edges()`` intervention, similar to ``cv.change_beta()`` but removes contacts entirely.


Version 0.28.1 (2020-04-19)
---------------------------
- Major refactor of transmission trees, including additional detail via ``sim.people.transtree.make_detailed()``.
- Counting of diagnoses before and after interventions on each timestep (allowing people to go into quarantine on the same day).
- Improved saving of people in scenarios, and updated keyword for sims (``sim.save(keep_people=True)``).


Version 0.28.0 (2020-04-19)
---------------------------
- Includes dynamic per-person viral load.
- Refactored data types.
- Changed how populations are handled, including adding a ``dynam_layer`` parameter to specify which layers are dynamic.
- Disease progression duration parameters were updated to be longer.
- Fixed bugs with quarantine.
- Fixed bug with hybrid school and work contacts.
- Changed contact tracing to be only for contacts with nonzero transmission.


Version 0.27.12 (2020-04-17)
----------------------------
- Caches Numba functions, reducing load time from 2.5 to 0.5 seconds.
- Pins Numba to 0.48, which is 10x faster than 0.49.
- Fixed issue with saving populations in scenarios.
- Refactored how populations are handled, removing ``use_layers`` parameter (use ``pop_type`` instead).
- Removed layer key from layer object, reducing total sim memory footprint by 3x.
- Improved handling of mismatches between loaded population layers and simulation parameters.
- Added custom key errors to handle multiline error messages.
- Fix several issues with probability-based testing.
- Changed how layer betas are applied (inside the sim rather than statically).
- Added more detail to the transmission tree.
- Refactored random population calculation, speeding up large populations (>100k) by a factor of 10.
- Added `documentation <https://institutefordiseasemodeling.github.io/covasim-docs/>`__.


Version 0.27.0 (2020-04-16)
---------------------------
-  Refactor calculations to be vector-based rather than object based.
-  Include factors for per-person viral load (transmissibility) and
   susceptibility.
-  Started a changelog (needless to say).
