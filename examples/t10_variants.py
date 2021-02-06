import covasim as cv


pars = {'n_strains': 2,
        'beta': [0.016, 0.035]}


sim = cv.Sim(pars=pars)
sim.run()

sim.plot_result('cum_infections_by_strain', do_show=False, do_save=True, fig_path='results/sim1_cum_infections_by_strain')
sim.plot_result('incidence_by_strain', label = ['strain1', 'strain2'], do_show=False, do_save=True, fig_path='results/sim1_incidence_by_strain')
sim.plot_result('prevalence_by_strain', label = ['strain1', 'strain2'], do_show=False, do_save=True, fig_path='results/sim1_prevalence_by_strain')

# Run sim with a single strain initially, then introduce a new strain that's more transmissible on day 10
pars = {'n_strains': 10, 'beta': [0.016]*10, 'max_strains': 11}
imports = cv.import_strain(day=10, n_imports=20, beta=0.05, rel_trans=1, rel_sus=1)
sim2 = cv.Sim(pars=pars, interventions=imports, label='With imported infections')
sim2.run()

sim2.plot_result('cum_infections_by_strain', do_show=False, do_save=True, fig_path='results/sim2_cum_infections_by_strain')
sim2.plot_result('incidence_by_strain', label = ['strain1', 'strain2'], do_show=False, do_save=True, fig_path='results/sim2_incidence_by_strain')
sim2.plot_result('prevalence_by_strain', label = ['strain1', 'strain2'], do_show=False, do_save=True, fig_path='results/sim2_prevalence_by_strain')

