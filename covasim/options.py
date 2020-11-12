'''
Define options for Covasim. Note that font size and verbose can be set directly, e.g.

    cv.options.font_size = 18

but if precision or parallel options are set, cv.options.apply() must be called
to recompile the Numba functions.
'''

import os

# Set the default font size -- if 0, use Matplotlib default
font_size = int(os.getenv('COVASIM_FONT_SIZE', 0))

# Set the font family
font_family = os.getenv('COVASIM_FONT_FAMILY', '')

# Set the default font size -- if 0, use Matplotlib default
dpi = int(os.getenv('COVASIM_DPI', 0))

# Set default verbosity
verbose = float(os.getenv('COVASIM_VERBOSE', 1.0))

# Set default arithmetic precision -- use 32-bit by default for speed and memory efficiency
precision = int(os.getenv('COVASIM_PRECISION', 32))

# Specify whether to allow parallel Numba calculation -- about 20% faster, but the random number stream becomes nondeterministic
numba_parallel = bool(int(os.getenv('COVASIM_NUMBA_PARALLEL', 0)))


def apply():
    '''
    Apply changes to Numba functions -- reloading modules is necessary for
    changes to propogate.

    **Example**::

        import covasim as cv
        cv.options.precision = 64
        cv.options.apply()
        sim = cv.Sim()
        sim.run()
        assert sim.people.rel_trans.dtype == np.float64
    '''
    import importlib
    import covasim as cv
    importlib.reload(cv.defaults)
    importlib.reload(cv.utils)
    importlib.reload(cv)
    return

