'''
Run the non-tutorial examples using pytest
'''

import pylab as pl
import sciris as sc
from pathlib import Path
import importlib.util as iu

pl.switch_backend('agg') # To avoid graphs from appearing -- if you want them, run the examples directly
cwd = Path(sc.thisdir(__file__))
examples_dir = cwd.joinpath('../examples')

def run_example(name):
    '''
    Execute an example py script as __main__

    Args:
        name (str): the filename without the .py extension
    '''
    spec = iu.spec_from_file_location("__main__", examples_dir.joinpath(f"{name}.py"))
    module = iu.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_run_scenarios():
    run_example("run_scenarios")

def test_run_sim():
    run_example("run_sim")

def test_simple():
    run_example("simple")


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    test_run_scenarios()
    test_run_sim()
    test_simple()

    sc.toc(T)
    print('Done.')
