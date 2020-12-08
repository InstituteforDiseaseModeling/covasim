'''
Demonstrate all interventions, taken from intervention docstrings
'''

#%% Housekeeping

import sciris as sc
import pylab as pl
import covasim as cv

do_plot = 1
verbose = 0


def test_all_interventions():
    ''' Test all interventions supported by Covasim '''

    pars = sc.objdict(
        pop_size     = 1e3,
        pop_infected = 10,
        pop_type     = 'hybrid',
        n_days       = 90,
    )


    #%% Define the interventions

    # 1. Dynamic pars
    i1a = cv.test_prob(start_day=5, symp_prob=0.3)
    i1b = cv.dynamic_pars({'beta':{'days':[40, 50], 'vals':[0.005, 0.015]}, 'rel_death_prob':{'days':30, 'vals':2.0}}) # Starting day 30, make diagnosed people stop transmitting


    # 2. Sequence
    i2 = cv.sequence(days=[15, 30, 45], interventions=[
                        cv.test_num(daily_tests=[20]*pars.n_days),
                        cv.test_prob(symp_prob=0.0),
                        cv.test_prob(symp_prob=0.2),
                    ])


    # 3. Change beta
    i3a = cv.change_beta([30, 50], [0.0, 1.0], layers='h')
    i3b = cv.change_beta([30, 40, 60], [0.0, 1.0, 0.5])


    # 4. Clip edges -- should match the change_beta scenarios -- note that intervention i07 was removed
    i4a = cv.clip_edges([30, 50], [0.0, 1.0], layers='h')
    i4b = cv.clip_edges([30, 40, 60], [0.0, 1.0, 0.5])


    # 5. Test number
    i5 = cv.test_num(daily_tests=[100, 100, 100, 0, 0, 0]*(pars.n_days//6))


    # 6. Test probability
    i6 = cv.test_prob(symp_prob=0.1)


    # 7. Contact tracing
    i7a = cv.test_prob(start_day=20, symp_prob=0.01, asymp_prob=0.0, symp_quar_prob=1.0, asymp_quar_prob=1.0, test_delay=0)
    i7b = cv.contact_tracing(start_day=20, trace_probs=dict(h=0.9, s=0.7, w=0.7, c=0.3), trace_time=dict(h=0, s=1, w=1, c=3))


    # 8. Combination
    i8a = cv.clip_edges(days=18, changes=0.0, layers='s') # Close schools
    i8b = cv.clip_edges(days=[20, 32, 45], changes=[0.7, 0.3, 0.9], layers=['w', 'c']) # Reduce work and community
    i8c = cv.test_prob(start_day=38, symp_prob=0.01, asymp_prob=0.0, symp_quar_prob=1.0, asymp_quar_prob=1.0, test_delay=2) # Start testing for TTQ
    i8d = cv.contact_tracing(start_day=40, trace_probs=dict(h=0.9, s=0.7, w=0.7, c=0.3), trace_time=dict(h=0, s=1, w=1, c=3)) # Start tracing for TTQ

    # 9. Vaccine
    i9a = cv.vaccine(days=20, prob=1.0, rel_sus=1.0, rel_symp=0.0)
    i9b = cv.vaccine(days=50, prob=1.0, rel_sus=0.0, rel_symp=0.0)

    #%% Create and run the simulations
    sims = sc.objdict()
    sims.dynamic      = cv.Sim(pars=pars, interventions=[i1a, i1b])
    sims.sequence     = cv.Sim(pars=pars, interventions=i2)
    sims.change_beta1 = cv.Sim(pars=pars, interventions=i3a)
    sims.clip_edges1  = cv.Sim(pars=pars, interventions=i4a) # Roughly equivalent to change_beta1
    sims.change_beta2 = cv.Sim(pars=pars, interventions=i3b)
    sims.clip_edges2  = cv.Sim(pars=pars, interventions=i4b) # Roughly equivalent to change_beta2
    sims.test_num     = cv.Sim(pars=pars, interventions=i5)
    sims.test_prob    = cv.Sim(pars=pars, interventions=i6)
    sims.tracing      = cv.Sim(pars=pars, interventions=[i7a, i7b])
    sims.combo        = cv.Sim(pars=pars, interventions=[i8a, i8b, i8c, i8d])
    sims.vaccine      = cv.Sim(pars=pars, interventions=[i9a, i9b])

    for key,sim in sims.items():
        sim.label = key
        sim.run(verbose=verbose)


    #%% Plotting
    if do_plot:
        for sim in sims.values():
            print(f'Running {sim.label}...')
            sim.plot()
            fig = pl.gcf()
            try:
                fig.axes[0].set_title(f'Simulation: {sim.label}')
            except:
                pass

    return



#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()

    test_all_interventions()

    sc.toc(T)
    print('Done.')