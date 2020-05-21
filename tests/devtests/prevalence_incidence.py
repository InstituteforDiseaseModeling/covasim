import covasim as cv

sim = cv.Sim(n_days=120, rescale=True, pop_scale=10)
sim.run()

to_plot = ['n_exposed', 'new_infections', 'prevalence', 'incidence']
sim.plot(to_plot=to_plot)
