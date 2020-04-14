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
fig_path  = 'results/testing_borderclosure.png'

def test_beds(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test of bed capacity estimation')

    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1

    basepars = {'pop_size': 1000}
    metapars = {'n_runs': n_runs}

    sim = cv.Sim()

    # Define the scenarios
    scenarios = {
        'baseline': {
          'name': 'No bed constraints',
          'pars': {
              'pop_infected': 100
          }
        },
        'bedconstraint': {
            'name': 'Only 10 beds available',
            'pars': {
                'pop_infected': 100,
                'n_beds': 10,
            }
        },
        'bedconstraint2': {
            'name': 'Only 1 bed available',
            'pars': {
                'pop_infected': 100,
                'n_beds': 1,
                # 'OR_no_treat': 10., # nb. scenarios cannot currently overwrite nested parameters
                # This prevents overwriting OR_no_treat due to recent refactoring but more generally
                # there are other nested parameters eg. all of those under pars['dur']
            }
        },
    }



    scens = cv.Scenarios(sim=sim, basepars=basepars, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        to_plot = sc.odict({
            'cum_deaths':   'Cumulative deaths',
           'bed_capacity': 'People needing beds / beds',
            'n_severe':     'Number of cases requiring hospitalization',
            'n_critical':   'Number of cases requiring ICU',
        })
        scens.plot(to_plot=to_plot, do_save=do_save, do_show=do_show, fig_path=fig_path)

    return scens


def test_borderclosure(do_plot=False, do_show=True, do_save=False, fig_path=None):
    sc.heading('Test effect of border closures')

    sc.heading('Setting up...')

    sc.tic()

    n_runs = 3
    verbose = 1

    basepars = {'pop_size': 1000}
    basepars = {'n_imports': 5}
    metapars = {'n_runs': n_runs}

    sim = cv.Sim()

    # Define the scenarios
    scenarios = {
        'baseline': {
            'name': 'No border closures',
            'pars': {
            }
        },
        'borderclosures_day1': {
          'name':'Close borders on day 1',
          'pars': {
              'interventions': [cv.dynamic_pars({'n_imports': {'days': 1, 'vals': 0}})]
          }
          },
        'borderclosures_day10': {
            'name': 'Close borders on day 10',
            'pars': {
                'interventions': [cv.dynamic_pars({'n_imports': {'days': 10, 'vals': 0}})]
            }
        },
    }

    scens = cv.Scenarios(sim=sim, basepars=basepars, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)

    if do_plot:
        scens.plot(do_save=do_save, do_show=do_show, fig_path=fig_path)

    return scens


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    bed_scens = test_beds(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_path)
    border_scens = test_borderclosure(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_path)

    sc.toc()


print('Done.')
