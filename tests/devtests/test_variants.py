import covasim as cv
import sciris as sc
import matplotlib
import matplotlib.pyplot as plt

do_plot   = 1
do_show   = 1
do_save   = 0


def test_multistrains(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Run basic sim with multiple strains')

    sc.heading('Setting up...')

    strain_labels = [
        'Strain 1: beta 0.016',
        'Strain 2: beta 0.035'
    ]

    immunity = [
        {'init_immunity':1., 'half_life':180, 'cross_factor':0.5},
        {'init_immunity':1., 'half_life':50,  'cross_factor':0.9}
    ]

    pars = {
        'n_strains': 2,
            'beta': [0.016, 0.035],
            'immunity': immunity
            }

    sim = cv.Sim(pars=pars)
    sim.run()

    if do_plot:
        plot_results(sim, key='incidence_by_strain', title='Multiple strains', labels=strain_labels)
    return sim


def test_importstrain_withcrossimmunity(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test introducing a new strain partway through a sim with full cross immunity')

    sc.heading('Setting up...')

    strain_labels = [
        'Strain 1: beta 0.016',
        'Strain 2: beta 0.05'
    ]
    # Run sim with several strains initially, then introduce a new strain that's more transmissible on day 10
    immunity = [
        {'init_immunity': 1., 'half_life': 180, 'cross_factor': 1},
    ]
    pars = {
        'n_strains': 1,
        'beta': [0.016],
        'immunity': immunity
    }
    imports = cv.import_strain(days=10, n_imports=30, beta=0.05, init_immunity=1, half_life=180, cross_factor=1)
    sim = cv.Sim(pars=pars, interventions=imports, label='With imported infections')
    sim.run()

    if do_plot:
        plot_results(sim, key='incidence_by_strain', title='Imported strain on day 10 (cross immunity)', labels=strain_labels)
    return sim

def test_importstrain_nocrossimmunity(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test introducing a new strain partway through a sim with cross immunity')

    sc.heading('Setting up...')

    strain_labels = [
        'Strain 1: beta 0.016',
        'Strain 2: beta 0.05'
    ]
    # Run sim with several strains initially, then introduce a new strain that's more transmissible on day 10
    immunity = [
        {'init_immunity': 1., 'half_life': 180, 'cross_factor': 0},
    ]
    pars = {
        'n_strains': 1,
        'beta': [0.016],
        'immunity': immunity
    }
    imports = cv.import_strain(days=10, n_imports=30, beta=0.05, init_immunity=1, half_life=180, cross_factor=0)
    sim = cv.Sim(pars=pars, interventions=imports, label='With imported infections')
    sim.run()

    if do_plot:
        plot_results(sim, key='incidence_by_strain', title='Imported strain on day 10 (no cross immunity)', labels=strain_labels)
    return sim

def plot_results(sim, key, title, labels=None):

    results = sim.results
    results_to_plot = results[key]

    # extract data for plotting
    x = sim.results['t']
    y = results_to_plot.values

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='Day of simulation', ylabel=results_to_plot.name, title=title)

    if labels is None:
        labels = [0]*len(y[0])
        for strain in range(len(y[0])):
            labels[strain] = f'Strain {strain +1}'
    ax.legend(labels)
    plt.show()

    return



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    sim1 = test_multistrains(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=None)
    sim2 = test_importstrain_withcrossimmunity(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=None)
    sim3 = test_importstrain_nocrossimmunity(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=None)

    sc.toc()


print('Done.')
