'''
Define options for Covasim. Note that font size and verbose can be set directly, e.g.

    cv.options.set('font_size', 18)

but if precision or parallel options are set, cv.options.apply() must be called
to recompile the Numba functions.
'''

import os
import sciris as sc

__all__ = ['options']


# Options acts like a class, but is actually an objdict for simplicity
options = sc.objdict()


# Set the default font size -- if 0, use Matplotlib default
options.font_size = int(os.getenv('COVASIM_FONT_SIZE', 0))

# Set the font family
options.font_family = os.getenv('COVASIM_FONT_FAMILY', '')

# Set the default font size -- if 0, use Matplotlib default
options.dpi = int(os.getenv('COVASIM_DPI', 0))

# Set default verbosity
options.verbose = float(os.getenv('COVASIM_VERBOSE', 1.0))

# Set default arithmetic precision -- use 32-bit by default for speed and memory efficiency
options.precision = int(os.getenv('COVASIM_PRECISION', 32))

# Specify whether to allow parallel Numba calculation -- about 20% faster, but the random number stream becomes nondeterministic
options.numba_parallel = bool(int(os.getenv('COVASIM_NUMBA_PARALLEL', 0)))

# Specify which keys require a reload
matplotlib_keys = ['font_size', 'font_family', 'dpi']
numba_keys = ['precision', 'numba_parallel']


def set_option(key=None, value=None, set_global=True, **kwargs):
    '''
    Set a parameter or parameters.

    Args:
        key (str): the parameter to modify
        value (varies): the value to specify
        set_global (bool): if true (default), sets plotting options globally (rather than just for Covasim)
        kwargs (dict): if supplied, set multiple key-value pairs

    Options are:

        - ``font_size``: The font size used for the plots (default: 10)
        - ``font_family``: The font family/face used for the plots (default: Open Sans)
        - ``dpi``: The overall DPI for the figure (default: 100)
        - ``verbose``: Default verbosity for simulations to use (default: 1)
        - ``precision``: The arithmetic to use in calculations (default: 32; other option is 64)
        - ``numba_parallel``: Whether to use parallel threads in Numba (default false, since gives non-reproducible results)

    **Examples**::

        cv.options.set('font_size', 18)
        cv.options.set(font_size=18, precision=64)
    '''
    if key is not None:
        kwargs = sc.mergedicts(kwargs, {key:value})
    reload_required = False
    for key,value in kwargs.items():
        if key not in options:
            keylist = [k for k in options.keys() if k not in ['set', 'apply']] # These are not keys
            keys = '\n'.join(keylist)
            errormsg = f'Option "{key}" not recognized; options are:\n{keys}'
            raise sc.KeyNotFoundError(errormsg)
        else:
            options[key] = value
            if key in numba_keys:
                reload_required = True
            if key in matplotlib_keys and set_global:
                set_matplotlib_global(key, value)
    if reload_required:
        options.apply()
    return


def set_matplotlib_global(key, value):
    ''' Set a global option for Matplotlib '''
    import pylab as pl
    if   key == 'font_size':   pl.rc('font', size=value)
    elif key == 'font_family': pl.rc('font', family=value)
    elif key == 'dpi':         pl.rc('figure', dpi=value)
    else: raise sc.KeyNotFoundError(f'Key {key} not found')
    return


def apply():
    '''
    Apply changes to Numba functions -- reloading modules is necessary for
    changes to propogate.

    **Example**::

        import covasim as cv
        cv.options.set(precision=64)
        sim = cv.Sim()
        sim.run()
        assert sim.people.rel_trans.dtype == np.float64
    '''
    print('Reloading Covasim so changes take effect...')
    import importlib
    import covasim as cv
    importlib.reload(cv.defaults)
    importlib.reload(cv.utils)
    importlib.reload(cv)
    return


# Add these here to be more accessible to the user
options.set = set_option
options.apply = apply