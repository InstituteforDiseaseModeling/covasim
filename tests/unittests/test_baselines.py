"""
Compare current results to baseline
"""

import sciris as sc
import covasim as cv

baseline_filename = 'baseline.json'


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


if __name__ == '__main__':

    new = test_baseline()

    print('Done.')