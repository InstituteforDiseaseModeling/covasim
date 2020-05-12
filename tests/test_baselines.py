"""
Compare current results to baseline
"""

import numpy as np
import pandas as pd
import sciris as sc
import covasim as cv

do_save = False
baseline_filename  = sc.thisdir(__file__, 'baseline.json')
benchmark_filename = sc.thisdir(__file__, 'benchmark.json')
parameters_filename = sc.thisdir(__file__, 'regression', f'parameters_v{cv.__version__}.json')
baseline_key = 'summary'


def save_baseline(do_save=do_save):
    ''' Refresh the baseline results '''
    print('Updating baseline values...')

    sim = cv.Sim(verbose=0)
    sim.run()
    if do_save:
        sim.to_json(filename=baseline_filename, keys=baseline_key)
        sim.export_pars(filename=parameters_filename)

    print('Done.')

    return sim


def test_baseline():
    ''' Compare the current default sim against the saved baseline '''

    # Load existing baseline
    baseline = sc.loadjson(baseline_filename)
    old = baseline[baseline_key]

    # Calculate new baseline
    sim = cv.Sim(verbose=0)
    sim.run()
    new = sim.summary

    # Compare keys
    errormsg = ''
    old_keys = set(old.keys())
    new_keys = set(new.keys())
    if old_keys != new_keys:
        errormsg = f"Keys don't match!\n"
        missing = old_keys - new_keys
        extra   = new_keys - old_keys
        if missing:
            errormsg += f'  Missing old keys: {missing}\n'
        if extra:
            errormsg += f'  Extra new keys: {extra}\n'

    mismatches = {}
    union = old_keys.union(new_keys)
    for key in new.keys(): # To ensure order
        if key in union:
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
        sim = cv.Sim(verbose=0)

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

    new  = test_baseline()
    json = test_benchmark(do_save=do_save)

    print('Done.')
