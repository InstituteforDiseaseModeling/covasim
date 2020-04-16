"""
Compare current results to baseline
"""

import sciris as sc
import covasim as cv

do_save = True
baseline_filename  = sc.thisdir(__file__, 'baseline.json')
benchmark_filename = sc.thisdir(__file__, 'benchmark.json')
baseline_key = 'summary'


def save_baseline(do_save=do_save):
    ''' Refresh the baseline results '''
    print('Updating baseline values...')

    sim = cv.Sim(verbose=0)
    sim.run()
    sim.to_json(filename=baseline_filename, keys=baseline_key)

    print('Done.')

    return sim


def test_baseline():
    ''' Compare the current default sim against the saved baseline '''

    # Load existing baseline
    filepath = sc.makefilepath(filename=baseline_filename, folder=sc.thisdir(__file__))
    baseline = sc.loadjson(filepath)
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
        extra = new_keys - old_keys
        if missing:
            errormsg += f'  Missing old keys: {missing}\n'
        if extra:
            errormsg += f'  Extra new keys: {extra}\n'

    mismatches = {}
    for key in old_keys.union(new_keys):
        old_val = old[key] if key in old else 'not present'
        new_val = new[key] if key in new else 'not present'
        if old_val != new_val:
            mismatches[key] = {'old': old_val, 'new': new_val}

    if len(mismatches):
        errormsg += '\nMismatches:\n'
        space = ' '*17
        for mkey,mval in mismatches.items():
            errormsg += f'  {mkey}:\n'
            errormsg += f'{space}old = {mval["old"]}\n'
            errormsg += f'{space}new = {mval["new"]}\n'

    # Raise an error if mismatches were found
    if errormsg:
        prefix = '\nThe following values have changed between the previous baseline and now!\n'
        prefix += 'If this is intentional, please rerun "update_baseline" and commit.\n\n'
        err = prefix + errormsg
        raise ValueError(err)
    else:
        print('Baseline matches')

    return new


def test_benchmark(do_save=do_save):
    ''' Compare benchmark performance '''

    print('Updating benchmark...')

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
            }

    if do_save:
        sc.savejson(filename=benchmark_filename, obj=json, indent=2)

    print('Done.')

    return json




if __name__ == '__main__':

    new  = test_baseline()
    json = test_benchmark(do_save=do_save)

    print('Done.')