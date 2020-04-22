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
fig_path  = 'results/testing_other.png'


def test_beta_edges(do_plot=False, do_show=True, do_save=False, fig_path=None):

    pars = dict(
        pop_size=2000,
        pop_infected=20,
        pop_type='hybrid',
        )

    start_day = 25 # Day to start the intervention
    end_day   = 40 # Day to end the intervention
    change    = 0.3 # Amount of change

    sims = sc.objdict()
    sims.b = cv.Sim(pars) # Beta intervention
    sims.e = cv.Sim(pars) # Edges intervention

    beta_interv = cv.change_beta(days=[start_day, end_day], changes=[change, 1.0])
    edge_interv = cv.clip_edges(start_day=start_day, end_day=end_day, change=change, verbose=True)
    sims.b.update_pars(interventions=beta_interv)
    sims.e.update_pars(interventions=edge_interv)

    for sim in sims.values():
        sim.run()
        if do_plot:
            sim.plot(do_save=do_save, do_show=do_show, fig_path=fig_path)
            sim.plot_result('r_eff')

    return sims


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
            'name': 'Only 50 beds available',
            'pars': {
                'pop_infected': 100,
                'n_beds': 50,
            }
        },
        'bedconstraint2': {
            'name': 'Only 10 beds available',
            'pars': {
                'pop_infected': 100,
                'n_beds': 10,
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

    sims = test_beta_edges(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_path)
    bed_scens = test_beds(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_path)
    border_scens = test_borderclosure(do_plot=do_plot, do_save=do_save, do_show=do_show, fig_path=fig_path)

    sc.toc()


print('Done.')
