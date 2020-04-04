'''
Testing the effect of testing interventions in Covasim
'''

#%% Imports and settings
import sciris as sc
import covasim as cv

do_plot   = 1
do_show   = 1
do_save   = 0
debug     = 1
fig_path  = 'results/testing_beds.png'

def test_beds(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test of bed capacity estimation')

    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1

    basepars = {'n': 1000}
    metapars = {'n_runs': n_runs}

    sim = cv.Sim()

    # Define the scenarios
    scenarios = {
        'baseline': {
          'name': 'No bed constraints',
          'pars': {
              'n_infected': 100
          }
        },
        'bedconstraint': {
            'name': 'Only 10 beds available',
            'pars': {
                'n_infected': 100,
                'n_beds': 10,
            }
        },
        'bedconstraint2': {
            'name': 'Only 1 bed available, people are 10x more likely to die if not hospitalized',
            'pars': {
                'n_infected': 100,
                'n_beds': 1,
                'OR_no_treat': 10.,
            }
        },
    }

    

    scens = cv.Scenarios(sim=sim, basepars=basepars, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        to_plot = sc.odict({
            'cum_deaths':   'Cumulative deaths',
#            'bed_capacity': 'People needing beds / beds',
            'n_severe':     'Number of cases requiring hospitalization',
            'n_critical':   'Number of cases requiring ICU',
        })
        scens.plot(to_plot=to_plot, do_save=do_save, do_show=do_show, fig_path=fig_path)

    return scens


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    scens = test_beds(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_path)

    sc.toc()


print('Done.')