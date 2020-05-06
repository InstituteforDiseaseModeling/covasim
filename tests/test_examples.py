'''
Run examples/*.py using pytest
'''

import importlib.util as iu
import os
from pathlib import Path

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
examples_dir = cwd.joinpath('../examples')

def run_example(name):
    """
    Execute an example py script as __main__
    :param name: the filename without the .py extension
    """
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
    sc.tic()
    test_run_scenarios()
    test_run_sim()
    test_simple()
    sc.toc()
    print('Done.')
