import covasim as cv
import sciris as sc

do_plot   = 1
do_show   = 0
do_save   = 1


def test_multistrains(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Run basic sim with multiple strains')

    sc.heading('Setting up...')

    pars = {'n_strains': 2,
            'beta': [0.016, 0.035]}

    sim = cv.Sim(pars=pars)
    sim.run()

    if do_plot:
        sim.plot_result('cum_infections_by_strain', do_show=do_show, do_save=do_save, fig_path='results/sim1_cum_infections_by_strain')
        sim.plot_result('incidence_by_strain', label=['strain1', 'strain2'], do_show=do_show, do_save=do_save, fig_path='results/sim1_incidence_by_strain')
        sim.plot_result('prevalence_by_strain', label=['strain1', 'strain2'], do_show=do_show, do_save=do_save, fig_path='results/sim1_prevalence_by_strain')

    return sim


def test_importstrain(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test introducing a new strain partway through a sim')

    sc.heading('Setting up...')

    # Run sim with several strains initially, then introduce a new strain that's more transmissible on day 10
    pars = {'n_strains': 10, 'beta': [0.016] * 10, 'max_strains': 11} # Checking here that increasing max_strains works
    imports = cv.import_strain(day=10, n_imports=20, beta=0.05, rel_trans=1, rel_sus=1)
    sim = cv.Sim(pars=pars, interventions=imports, label='With imported infections')
    sim.run()

    if do_plot:
        sim.plot_result('cum_infections_by_strain', do_show=do_show, do_save=do_save, fig_path='results/sim2_cum_infections_by_strain')
        sim.plot_result('incidence_by_strain', label=['strain1', 'strain2'], do_show=do_show, do_save=do_save, fig_path='results/sim2_incidence_by_strain')
        sim.plot_result('prevalence_by_strain', label=['strain1', 'strain2'], do_show=do_show, do_save=do_save, fig_path='results/sim2_prevalence_by_strain')

    return sim



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    sim1 = test_multistrains(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=None)
    #sim2 = test_importstrain(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=None)

    sc.toc()


print('Done.')
