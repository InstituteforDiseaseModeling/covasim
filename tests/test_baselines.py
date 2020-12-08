"""
Compare current results to baseline
"""

import numpy as np
import pandas as pd
import sciris as sc
import covasim as cv

do_plot = 1
do_save = 0
baseline_filename  = sc.thisdir(__file__, 'baseline.json')
benchmark_filename = sc.thisdir(__file__, 'benchmark.json')
parameters_filename = sc.thisdir(cv.__file__, 'regression', f'pars_v{cv.__version__}.json')


def make_sim(use_defaults=False, do_plot=False, **kwargs):
    '''
    Define a default simulation for testing the baseline -- use hybrid and include
    interventions to increase coverage. If run directly (not via pytest), also
    plot the sim by default.
    '''

    # Define the interventions
    cb = cv.change_beta(days=40, changes=0.5)
    tp = cv.test_prob(start_day=20, symp_prob=0.1, asymp_prob=0.01)
    ct = cv.contact_tracing(trace_probs=0.3, start_day=50)

    # Define the parameters
    pars = dict(
        pop_size      = 20e3,         # Population size
        pop_infected  = 100,          # Number of initial infections -- use more for increased robustness
        pop_type      = 'hybrid',     # Population to use -- "hybrid" is random with household, school,and work structure
        n_days        = 60,           # Number of days to simulate
        verbose       = 0,            # Don't print details of the run
        rand_seed     = 2,            # Set a non-default seed
        interventions = [cb, tp, ct], # Include the most common interventions
    )
    pars = sc.mergedicts(pars, kwargs)

    # Create the sim
    if use_defaults:
        sim = cv.Sim()
    else:
        sim = cv.Sim(pars)

    # Optionally plot
    if do_plot:
        s2 = sim.copy()
        s2.run()
        s2.plot()

    return sim


def save_baseline():
    '''
    Refresh the baseline results. This function is not called during standard testing,
    but instead is called by the update_baseline script.
    '''

    print('Updating baseline values...')

    # Export default parameters
    s1 = make_sim(use_defaults=True)
    s1.export_pars(filename=parameters_filename)

    # Export results
    s2 = make_sim(use_defaults=False)
    s2.run()
    s2.to_json(filename=baseline_filename, keys='summary')

    print('Done.')

    return


def test_baseline():
    ''' Compare the current default sim against the saved baseline '''

    # Load existing baseline
    baseline = sc.loadjson(baseline_filename)
    old = baseline['summary']

    # Calculate new baseline
    sim = make_sim()
    sim.run()
    new = sim.summary

    # Compare keys
    errormsg = ''
    old_keys = set(old.keys())
    new_keys = set(new.keys())
    if old_keys != new_keys:
        errormsg = "Keys don't match!\n"
        missing = list(old_keys - new_keys)
        extra   = list(new_keys - old_keys)
        if missing:
            errormsg += f'  Missing old keys: {missing}\n'
        if extra:
            errormsg += f'  Extra new keys: {extra}\n'

    mismatches = {}
    for key in new.keys(): # To ensure order
        if key in old_keys: # If a key is missing, don't count it as a mismatch
            old_val = old[key] if key in old else 'not present'
            new_val = new[key] if key in new else 'not present'
            if old_val != new_val:
                mismatches[key] = {'old': old_val, 'new': new_val}

    if len(mismatches):
        errormsg = '\nThe following values have changed from the previous baseline!\n'
        errormsg += 'If this is intentional, please rerun "tests/update_baseline" and commit.\n'
        errormsg += 'Mismatches:\n'
        df = pd.DataFrame.from_dict(mismatches).transpose()
        diff   = []
        ratio  = []
        change = []
        small_change = 1e-3 # Define a small change, e.g. a rounding error
        for mdict in mismatches.values():
            old = mdict['old']
            new = mdict['new']
            if sc.isnumber(new) and sc.isnumber(old) and old>0:
                this_diff  = new - old
                this_ratio = new/old
                abs_ratio  = max(this_ratio, 1.0/this_ratio)

                # Set the character to use
                if abs_ratio<small_change:
                    change_char = '≈'
                elif new > old:
                    change_char = '↑'
                elif new < old:
                    change_char = '↓'
                else:
                    errormsg = f'Could not determine relationship between old={old} and new={new}'
                    raise ValueError(errormsg)

                # Set how many repeats it should have
                repeats = 1
                if abs_ratio >= 1.1:
                    repeats = 2
                if abs_ratio >= 2:
                    repeats = 3
                if abs_ratio >= 10:
                    repeats = 4

                this_change = change_char*repeats
            else:
                this_diff   = np.nan
                this_ratio  = np.nan
                this_change = 'N/A'

            diff.append(this_diff)
            ratio.append(this_ratio)
            change.append(this_change)

        df['diff']   = diff
        df['ratio']  = ratio
        for col in ['old', 'new', 'diff', 'ratio']:
            df[col] = df[col].round(decimals=3)
        df['change'] = change
        errormsg += str(df)

    # Raise an error if mismatches were found
    if errormsg:
        raise ValueError(errormsg)
    else:
        print('Baseline matches')

    return new


def test_benchmark(do_save=do_save):
    ''' Compare benchmark performance '''

    print('Running benchmark...')
    previous = sc.loadjson(benchmark_filename)

    repeats = 5
    t_inits = []
    t_runs  = []

    def normalize_performance():
        ''' Normalize performance across CPUs -- simple Numpy calculation '''
        t_bls = []
        bl_repeats = 5
        n_outer = 10
        n_inner = 1e6
        for r in range(bl_repeats):
            t0 = sc.tic()
            for i in range(n_outer):
                a = np.random.random(int(n_inner))
                b = np.random.random(int(n_inner))
                a*b
            t_bl = sc.toc(t0, output=True)
            t_bls.append(t_bl)
        t_bl = min(t_bls)
        reference = 0.112 # Benchmarked on an Intel i9-8950HK CPU @ 2.90GHz
        ratio = reference/t_bl
        return ratio


    # Test CPU performance before the run
    r1 = normalize_performance()

    # Do the actual benchmarking
    for r in range(repeats):

        # Create the sim
        sim = make_sim(verbose=0)

        # Time initialization
        t0 = sc.tic()
        sim.initialize()
        t_init = sc.toc(t0, output=True)

        # Time running
        t0 = sc.tic()
        sim.run()
        t_run = sc.toc(t0, output=True)

        # Store results
        t_inits.append(t_init)
        t_runs.append(t_run)

    # Test CPU performance after the run
    r2 = normalize_performance()
    ratio = (r1+r2)/2
    t_init = min(t_inits)*ratio
    t_run  = min(t_runs)*ratio

    # Construct json
    n_decimals = 3
    json = {'time': {
                'initialize': round(t_init, n_decimals),
                'run':        round(t_run,  n_decimals),
                },
            'parameters': {
                'pop_size': sim['pop_size'],
                'pop_type': sim['pop_type'],
                'n_days':   sim['n_days'],
                },
            'cpu_performance': ratio,
            }

    print('Previous benchmark:')
    sc.pp(previous)

    print('\nNew benchmark:')
    sc.pp(json)

    if do_save:
        sc.savejson(filename=benchmark_filename, obj=json, indent=2)

    print('Done.')

    return json



if __name__ == '__main__':

    make_sim(do_plot=do_plot)
    json = test_benchmark(do_save=do_save) # Run this first so benchmarking is available even if results are different
    new  = test_baseline()

    print('Done.')
