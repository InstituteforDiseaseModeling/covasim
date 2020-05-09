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
    i00 = cv.test_prob(start_day=5, symp_prob=0.3)
    i01 = cv.dynamic_pars({'beta':{'days':[40, 50], 'vals':[0.005, 0.015]}, 'diag_factor':{'days':30, 'vals':0.0}}) # Starting day 30, make diagnosed people stop transmitting


    # 2. Sequence
    i02 = cv.sequence(days=[15, 30, 45], interventions=[
                        cv.test_num(daily_tests=[20]*pars.n_days),
                        cv.test_prob(symp_prob=0.0),
                        cv.test_prob(symp_prob=0.2),
                    ])


    # 3. Change beta
    i03 = cv.change_beta([30, 50], [0.0, 1], layers='h')
    i04 = cv.change_beta([30, 40, 60], [0.0, 1.0, 0.5])


    # 4. Clip edges -- should match the change_beta scenarios
    i05 = cv.clip_edges(start_day=30, end_day=50, change={'h':0.0})
    i06 = cv.clip_edges(start_day=30, end_day=40, change=0.0)
    i07 = cv.clip_edges(start_day=60, end_day=None, change=0.5)


    # 5. Test number
    i08 = cv.test_num(daily_tests=[100, 100, 100, 0, 0, 0]*(pars.n_days//6))


    # 6. Test probability
    i09 = cv.test_prob(symp_prob=0.1)


    # 7. Contact tracing
    i10 = cv.test_prob(start_day=20, symp_prob=0.01, asymp_prob=0.0, symp_quar_prob=1.0, asymp_quar_prob=1.0, test_delay=0)
    i11 = cv.contact_tracing(start_day=20, trace_probs=dict(h=0.9, s=0.7, w=0.7, c=0.3), trace_time=dict(h=0, s=1, w=1, c=3))


    # 8. Combination
    i12 = cv.clip_edges(start_day=18, change={'s':0.0}) # Close schools
    i13 = cv.clip_edges(start_day=20, end_day=32,   change={'w':0.7, 'c':0.7}) # Reduce work and community
    i14 = cv.clip_edges(start_day=32, end_day=45,   change={'w':0.3, 'c':0.3}) # Reduce work and community more
    i15 = cv.clip_edges(start_day=45, end_day=None, change={'w':0.9, 'c':0.9}) # Reopen work and community more
    i16 = cv.test_prob(start_day=38, symp_prob=0.01, asymp_prob=0.0, symp_quar_prob=1.0, asymp_quar_prob=1.0, test_delay=2) # Start testing for TTQ
    i17 = cv.contact_tracing(start_day=40, trace_probs=dict(h=0.9, s=0.7, w=0.7, c=0.3), trace_time=dict(h=0, s=1, w=1, c=3)) # Start tracing for TTQ


    #%% Create and run the simulations
    sims = sc.objdict()
    sims.dynamic      = cv.Sim(pars=pars, interventions=[i00, i01])
    sims.sequence     = cv.Sim(pars=pars, interventions=i02)
    sims.change_beta1 = cv.Sim(pars=pars, interventions=i03)
    sims.clip_edges1  = cv.Sim(pars=pars, interventions=i05) # Roughly equivalent to change_beta1
    sims.change_beta2 = cv.Sim(pars=pars, interventions=i04)
    sims.clip_edges2  = cv.Sim(pars=pars, interventions=[i06, i07]) # Roughly euivalent to change_beta2
    sims.test_num     = cv.Sim(pars=pars, interventions=i08)
    sims.test_prob    = cv.Sim(pars=pars, interventions=i09)
    sims.tracing      = cv.Sim(pars=pars, interventions=[i10, i11])
    sims.combo        = cv.Sim(pars=pars, interventions=[i12, i13, i14, i15, i16, i17])

    for key,sim in sims.items():
        sim.label = key
        sim.run(verbose=verbose)


    #%% Plotting
    if do_plot:
        for sim in sims.values():
            print(f'Running {sim.label}...')
            sim.plot()
            fig = pl.gcf()
            fig.axes[0].set_title(f'Simulation: {sim.label}')

    return



#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()

    test_all_interventions()

    sc.toc(T)
    print('Done.')