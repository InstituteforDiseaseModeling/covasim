import covasim as cv


pars = {'n_strains': 1,
        'beta': [0.016]}
sim = cv.Sim(pars=pars)
sim.run()

sim2 = cv.Sim()
sim2.run()
# sim.plot_result('cum_infections_by_strain', do_show=True)
# sim.plot_result('incidence_by_strain', label = ['strain1', 'strain2'], do_show=True)
# sim.plot_result('prevalence_by_strain', label = ['strain1', 'strain2'], do_show=True)

print('done')