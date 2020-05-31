#!/usr/bin/env pytest
# Run all dev tests. Note: these are not guaranteed to work.

import sciris as sc
import runpy
import pytest

scripts = sc.getfilelist(ext='py')

script_str= "\n".join(scripts)
print(f'Running tests on:\n{script_str}...')

@pytest.mark.parametrize('script', scripts)
def test_script_execution(script):
    runpy.run_path(script)