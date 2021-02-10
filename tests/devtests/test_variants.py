import covasim as cv
import sciris as sc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


do_plot   = 1
do_show   = 0
do_save   = 1


def test_multistrains(do_plot=False, do_show=True, do_save=False):
    sc.heading('Run basic sim with multiple strains')

    sc.heading('Setting up...')

    strain_labels = [
        'Strain 1: beta 0.016',
        'Strain 2: beta 0.035'
    ]

    immunity = [
        {'init_immunity':1., 'half_life':180},
        {'init_immunity':1., 'half_life':50}
    ]

    cross_immunity = np.array([[np.nan, 0.5],[0.5, np.nan]])

    pars = {
        'n_strains': 2,
        'beta': [0.016, 0.035],
        'immunity': immunity,
        'cross_immunity':cross_immunity
    }

    sim = cv.Sim(pars=pars)
    sim.run()

    if do_plot:
        plot_results(sim, key='incidence_by_strain', title='Multiple strains', labels=strain_labels, do_show=do_show, do_save=do_save)
    return sim


def test_importstrain_args():
    sc.heading('Test flexibility of arguments for the import strain "intervention"')

    # Run sim with a single strain initially, then introduce a new strain that's more transmissible on day 10
    immunity = [
        {'init_immunity': 1., 'half_life': 180, 'cross_factor': 1},
    ]
    pars = {
        'n_strains': 1,
        'beta': [0.016],
        'immunity': immunity
    }

    # All these should run
    #imports = cv.import_strain(days=50, beta=0.03)
    #imports = cv.import_strain(days=[10, 50], beta=0.03)
    #imports = cv.import_strain(days=50, beta=[0.03, 0.05])
    #imports = cv.import_strain(days=[10, 20], beta=[0.03, 0.05])
    #imports = cv.import_strain(days=50, beta=[0.03, 0.05, 0.06])
    #imports = cv.import_strain(days=[10, 20], n_imports=[5, 10], beta=[0.03, 0.05], init_immunity=[1, 1],
    #                          half_life=[180, 180], cross_factor=[0, 0])
    #imports = cv.import_strain(days=[10, 50], beta=0.03, cross_factor=[0.4, 0.6])
    #imports = cv.import_strain(days=['2020-04-01', '2020-05-01'], beta=0.03)

    # This should fail
    #imports = cv.import_strain(days=[20, 50], beta=[0.03, 0.05, 0.06])

    sim = cv.Sim(pars=pars, interventions=imports)
    sim.run()


    return sim


def test_importstrain_withcrossimmunity(do_plot=False, do_show=True, do_save=False):
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
    imports = cv.import_strain(days=30, n_imports=30, beta=0.05, init_immunity=1, half_life=180, cross_factor=1)
    sim = cv.Sim(pars=pars, interventions=imports, label='With imported infections')
    sim.run()

    if do_plot:
        plot_results(sim, key='incidence_by_strain', title='Imported strain on day 30 (cross immunity)', labels=strain_labels, do_show=do_show, do_save=do_save)
    return sim

def test_importstrain_nocrossimmunity(do_plot=False, do_show=True, do_save=False):
    sc.heading('Test introducing a new strain partway through a sim with cross immunity')

    sc.heading('Setting up...')


    # Run sim with several strains initially, then introduce a new strain that's more transmissible on day 10
    immunity = [
        {'init_immunity': 1., 'half_life': 180, 'cross_factor': 0},
    ]
    pars = {
        'n_strains': 1,
        'beta': [0.016],
        'immunity': immunity
    }
    imports = cv.import_strain(days=[10, 20], n_imports=[10, 20], beta=[0.035, 0.05], init_immunity=[1, 1],
                               half_life=[180, 180], cross_factor=[0, 0])
    sim = cv.Sim(pars=pars, interventions=imports, label='With imported infections')
    sim.run()

    strain_labels = [
        'Strain 1: beta 0.016',
        'Strain 2: beta 0.035, 10 imported day 10',
        'Strain 3: beta 0.05, 20 imported day 20'
    ]

    if do_plot:
        plot_results(sim, key='incidence_by_strain', title='Imported strains', labels=strain_labels, do_show=do_show, do_save=do_save)
    return sim

def plot_results(sim, key, title, do_show=True, do_save=False, labels=None):

    results = sim.results
    results_to_plot = results[key]

    # extract data for plotting
    x = sim.results['t']
    y = results_to_plot.values
    y = np.flip(y, 1)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='Day of simulation', ylabel=results_to_plot.name, title=title)

    if labels is None:
        labels = [0]*len(y[0])
        for strain in range(len(y[0])):
            labels[strain] = f'Strain {strain +1}'
    ax.legend(labels)
    if do_show:
        plt.show()
    if do_save:
        cv.savefig(f'results/{title}.png')

    return



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    sim1 = test_multistrains(do_plot=do_plot, do_save=do_save, do_show=do_show)
    # sim2 = test_importstrain_withcrossimmunity(do_plot=do_plot, do_save=do_save, do_show=do_show)
    # sim3 = test_importstrain_nocrossimmunity(do_plot=do_plot, do_save=do_save, do_show=do_show)
    # sim4 = test_importstrain_args()

    sc.toc()


print('Done.')
