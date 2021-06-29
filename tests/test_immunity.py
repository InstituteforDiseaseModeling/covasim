'''
Tests for immune waning, variants, and vaccine intervention.
'''

#%% Imports and settings
import sciris as sc
import covasim as cv
import pandas as pd
import pylab as pl
import numpy as np

do_plot = 1
cv.options.set(interactive=False) # Assume not running interactively

# Shared parameters across simulations
base_pars = dict(
    pop_size = 1e3,
    verbose  = -1,
)


#%% Define the tests

def test_states():
    ''' Test state consistency against state_diagram.xlsx '''

    filename = 'state_diagram.xlsx'
    sheets   = ['Without waning', 'With waning']
    indexcol = 'In ↓ you can be →'

    # Load state diagram
    dfs = sc.odict()
    for sheet in sheets:
        dfs[sheet] = pd.read_excel(filename, sheet_name=sheet)
        dfs[sheet] = dfs[sheet].set_index(indexcol)

    # Create and run simulation
    for use_waning in [False, True]:
        sc.heading(f'Testing state consistency with waning = {use_waning}')
        df = dfs[use_waning] # Different states are possible with or without waning

        # Parameters chosen to be midway through the sim so as few states as possible are empty
        pars = dict(
            pop_size = 1e3,
            pop_infected = 20,
            n_days = 70,
            use_waning = use_waning,
            verbose = 0,
            interventions = [
                cv.test_prob(symp_prob=0.4, asymp_prob=0.01),
                cv.contact_tracing(trace_probs=0.1),
                cv.simple_vaccine(days=60, prob=0.1)
            ]
        )
        sim = cv.Sim(pars).run()
        ppl = sim.people

        # Check states
        errormsg = ''
        states = df.columns.values.tolist()
        for s1 in states:
            for s2 in states:
                if s1 != s2:
                    relation = df.loc[s1, s2] # e.g. df.loc['susceptible', 'exposed']
                    print(f'Checking {s1:13s} → {s2:13s} = {relation:2n} ... ', end='')
                    inds     = cv.true(ppl[s1])
                    n_inds   = len(inds)
                    vals2    = ppl[s2][inds]
                    is_true  = cv.true(vals2)
                    is_false = cv.false(vals2)
                    n_true   = len(is_true)
                    n_false  = len(is_false)
                    if relation == 1 and n_true != n_inds:
                        errormsg = f'Being {s1}=True implies {s2}=True, but only {n_true}/{n_inds} people are'
                        print(f'× {n_true}/{n_inds} error!')
                    elif relation == -1 and n_false != n_inds:
                        errormsg = f'Being {s1}=True implies {s2}=False, but only {n_false}/{n_inds} people are'
                        print(f'× {n_false}/{n_inds} error!')
                    else:
                        print(f'✓ {n_true}/{n_inds}')
                    if errormsg:
                        raise RuntimeError(errormsg)

    return


def test_waning(do_plot=False):
    sc.heading('Testing with and without waning')
    msims = dict()

    for rescale in [0, 1]:
        print(f'Checking with rescale = {rescale}...')

        # Define parameters specific to this test
        pars = dict(
            n_days    = 90,
            beta      = 0.01,
        )

        # Optionally include rescaling
        if rescale:
            pars.update(
                pop_scale      = 10,
                rescale_factor = 2.0, # Use a large rescale factor to make differences more obvious
            )

        # Run the simulations and pull out the results
        s0 = cv.Sim(base_pars, **pars, use_waning=False, label='No waning').run()
        s1 = cv.Sim(base_pars, **pars, use_waning=True, label='With waning').run()
        res0 = s0.summary
        res1 = s1.summary
        msim = cv.MultiSim([s0,s1])
        msims[rescale] = msim

        # Check results
        for key in ['n_susceptible', 'cum_infections', 'cum_reinfections', 'pop_nabs', 'pop_protection', 'pop_symp_protection']:
            v0 = res0[key]
            v1 = res1[key]
            print(f'Checking {key:20s} ... ', end='')
            assert v1 > v0, f'Expected {key} to be higher with waning ({v1}) than without ({v0})'
            print(f'✓ ({v1} > {v0})')

        # Optionally plot
        if do_plot:
            msim.plot('overview-variant', rotation=30)

    return msims


def test_variants(do_plot=False):
    sc.heading('Testing variants...')

    b117 = cv.variant('b117',         days=10, n_imports=20)
    p1   = cv.variant('sa variant',   days=20, n_imports=20)
    cust = cv.variant(label='Custom', days=40, n_imports=20, variant={'rel_beta': 2, 'rel_symp_prob': 1.6})
    sim  = cv.Sim(base_pars, use_waning=True, variants=[b117, p1, cust])
    sim.run()

    if do_plot:
        sim.plot('overview-variant')

    return sim


def test_vaccines(do_plot=False):
    sc.heading('Testing vaccines...')

    p1 = cv.variant('sa variant',   days=20, n_imports=20)
    pfizer = cv.vaccinate_prob(vaccine='pfizer', days=30)
    sim  = cv.Sim(base_pars, use_waning=True, variants=p1, interventions=pfizer)
    sim.run()

    if do_plot:
        sim.plot('overview-variant')

    return sim


def test_vaccines_sequential(do_plot=False):
    sc.heading('Testing sequential vaccine...')

    p1 = cv.variant('sa variant',   days=20, n_imports=20)
    def age_sequence(people): return np.argsort(-people.age)

    n_doses = []
    pfizer = cv.vaccinate_num(vaccine='pfizer', sequence=age_sequence, num_doses=lambda sim: sim.t)
    sim  = cv.Sim(base_pars, rescale=False, use_waning=True, variants=p1, interventions=pfizer, analyzers=lambda sim: n_doses.append(sim.people.vaccinations.copy()))
    sim.run()

    n_doses = np.array(n_doses)

    if do_plot:
        fully_vaccinated = (n_doses == 2).sum(axis=1)
        first_dose = (n_doses == 1).sum(axis=1)
        pl.stackplot(sim.tvec, first_dose, fully_vaccinated)

        # Stacked bars by 10 year age

        # At the end of the simulation
        df = pd.DataFrame(n_doses.T)
        df['age_bin'] = np.digitize(sim.people.age,np.arange(0,100,10))
        df['fully_vaccinated'] = df[60]==2
        df['first_dose'] = df[60]==1
        df['unvaccinated'] = df[60]==0
        out = df.groupby('age_bin').sum()
        out[["unvaccinated", "first_dose","fully_vaccinated"]].plot(kind="bar", stacked=True)

        # Part-way through the simulation
        df = pd.DataFrame(n_doses.T)
        df['age_bin'] = np.digitize(sim.people.age,np.arange(0,100,10))
        df['fully_vaccinated'] = df[40]==2
        df['first_dose'] = df[40]==1
        df['unvaccinated'] = df[40]==0
        out = df.groupby('age_bin').sum()
        out[["unvaccinated", "first_dose","fully_vaccinated"]].plot(kind="bar", stacked=True)

    return sim


def test_two_vaccines(do_plot=False):
    sc.heading('Testing nab decay in simulation...')

    p1 = cv.variant('sa variant',   days=20, n_imports=0)

    nabs = []
    vac1 = cv.vaccinate_num(vaccine='pfizer', sequence=[0], num_doses=1)
    vac2 = cv.vaccinate_num(vaccine='jj', sequence=[1], num_doses=1)

    sim  = cv.Sim(base_pars, n_days=1000, pop_size=2, pop_infected=0, rescale=False, use_waning=True, variants=p1, interventions=[vac1, vac2], analyzers=lambda sim: nabs.append(sim.people.nab.copy()))
    sim.run()

    if do_plot:
        nabs = np.array(nabs)
        print(sim.people.peak_nab)
        pl.figure()
        pl.plot(nabs)


def test_decays(do_plot=False):
    sc.heading('Testing decay parameters...')

    n = 300
    x = pl.arange(n)

    pars = sc.objdict(

        nab_growth_decay = dict(
            func = cv.immunity.nab_growth_decay,
            length = n,
            growth_time = 22,
            decay_rate1 = np.log(2)/100,
            decay_time1 = 250,
            decay_rate2 = np.log(2)/3650,
            decay_time2 = 365,
        ),

        nab_decay = dict(
            func = cv.immunity.nab_decay,
            length = n,
            decay_rate1 = 0.05,
            decay_time1 = 100,
            decay_rate2 = 0.002,
        ),

        exp_decay = dict(
            func = cv.immunity.exp_decay,
            length = n,
            init_val = 0.8,
            half_life = 100,
            delay = 20,
        ),

        linear_decay = dict(
            func = cv.immunity.linear_decay,
            length = n,
            init_val = 0.8,
            slope = 0.01,
        ),

        linear_growth = dict(
            func = cv.immunity.linear_growth,
            length = n,
            slope = 0.01,
        ),
    )

    # Calculate all the delays
    res = sc.objdict()
    for key,par in pars.items():
        func = par.pop('func')
        res[key] = func(**par)

    if do_plot:
        pl.figure(figsize=(12,8))
        for key,y in res.items():
            pl.semilogy(x, np.cumsum(y), label=key, lw=3, alpha=0.7)
        pl.legend()
        pl.show()

    res.x = x

    return res


#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    cv.options.set(interactive=do_plot)
    T = sc.tic()

    sim1   = test_states()
    msims1 = test_waning(do_plot=do_plot)
    sim2   = test_variants(do_plot=do_plot)
    sim3   = test_vaccines(do_plot=do_plot)
    sim4   = test_vaccines_sequential(do_plot=do_plot)
    sim5   = test_two_vaccines(do_plot=do_plot)
    res    = test_decays(do_plot=do_plot)

    sc.toc(T)
    print('Done.')
