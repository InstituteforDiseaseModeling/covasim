import covasim as cv


pars = {'n_strains': 1,
        'beta': [0.016]}

imports = cv.import_strain(day=10, n_imports=[5], beta=[0.05], rel_trans=[1], rel_sus=[1])
# sim = cv.Sim(pars=pars)
# sim.run()
#
# sim.plot_result('cum_infections_by_strain', do_show=True)
# sim.plot_result('incidence_by_strain', label = ['strain1', 'strain2'], do_show=True)
# sim.plot_result('prevalence_by_strain', label = ['strain1', 'strain2'], do_show=True)

sim2 = cv.Sim(pars=pars, interventions=imports, label='With imported infections')
sim2.run()

sim2.plot_result('cum_infections_by_strain', do_show=True)
sim2.plot_result('incidence_by_strain', label = ['strain1', 'strain2'], do_show=True)
sim2.plot_result('prevalence_by_strain', label = ['strain1', 'strain2'], do_show=True)

