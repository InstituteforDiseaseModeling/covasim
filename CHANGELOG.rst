What's new
==========

All notable changes to the codebase are documented in this file. Note: in many cases, changes from multiple patch versions are grouped together, so numbering will not be strictly consecutive.


Version 1.0.0 (2020-05-08)
--------------------------
- Official release of Covasim.
- Made scenario and simulation plotting more flexible: ``to_plot`` can now simply be a list of results keys, e.g. ``cum_deaths``.
- Added additional tests, increasing test coverage from 67% to 92%.
- Fixed bug in ``cv.save()``.
- Added ``reset()`` to MultiSim that undoes a ``reduce()`` or ``combine()`` call.
- General code cleaning: made exceptions raised more consistent, removed unused functions, etc.
- GitHub info: PR `487 <https://github.com/amath-idm/covasim/pull/487>`__, previous head ``c8ca32d``


Version 0.32.1 (2020-05-06)
---------------------------
- Allow ``until`` to be a date, e.g. ``sim.run(until='2020-05-06')``.
- Added ``ipywidgets`` dependency since otherwise the webapp breaks due to a `bug <https://github.com/plotly/plotly.py/issues/2443>`__ with the latest Plotly version (4.7). 


Version 0.32.0 (2020-05-05)
---------------------------
- Changed the edges of the contact network from being directed to undirected, halving the amount of memory required and making contact tracing and edge clipping more realistic.
- Added comorbidities to the prognoses parameters.
- GitHub info: PR `482 <https://github.com/amath-idm/covasim/pull/482>`__, previous head ``029585f``


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
- GitHub info: PR `480 <https://github.com/amath-idm/covasim/pull/480>`__, previous head ``c7171f8``


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
