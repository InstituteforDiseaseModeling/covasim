"""
Compare current results to baseline
"""

import sciris as sc
import covasim as cv

do_save = True
baseline_filename  = 'baseline.json'
benchmark_filename = 'benchmark.json'


def test_baseline():
    ''' Compare the current default sim against the saved baseline '''

    # Load existing baseline
    filepath = sc.makefilepath(filename=baseline_filename, folder=sc.thisdir(__file__))
    baseline = sc.loadjson(filepath)
    old = baseline['summary']

    # Calculate new baseline
    sim = cv.Sim(verbose=0)
    sim.run()
    new = sim.summary

    # Compare keys
    old_keys = set(old.keys())
    new_keys = set(new.keys())
    if old_keys != new_keys:
        errormsg = f"Keys don't match; old: {old_keys}; new: {new_keys}"
        raise KeyError(errormsg)

    mismatches = {}
    for key in old.keys():
        old_val = old[key]
        new_val = new[key]
        if old_val != new_val:
            mismatches[key] = {'old': old_val, 'new': new_val}

    if len(mismatches):
        errormsg = '\nThe following values have changed between old and new!\n'
        errormsg += 'Please rerun "update_baseline" if this is intentional.\n'
        errormsg += 'Mismatches:\n'
        space = ' '*15
        for mkey,mval in mismatches.items():
            errormsg += f'{mkey}:\n'
            errormsg += f'{space}old = {mval["old"]}\n'
            errormsg += f'{space}new = {mval["new"]}\n'
        raise ValueError(errormsg)
    else:
        print('Baseline matches')

    return new


def test_benchmark(do_save=True):
    ''' Compare benchmark performance '''

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

    return json




if __name__ == '__main__':

    new  = test_baseline()
    json = test_benchmark()

    print('Done.')