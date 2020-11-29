'''
Define options for Covasim, mostly plotting and Numba options. All options should
be set using set(), e.g.::

    cv.options.set(font_size=18)
'''

import os
import sciris as sc

__all__ = ['options']


def set_default_options():
    ''' Set the default options for Covasim '''

    # Options acts like a class, but is actually an objdict for simplicity
    options = sc.objdict()

    # Set the default font size -- if 0, use Matplotlib default
    options.font_size = int(os.getenv('COVASIM_FONT_SIZE', 0))

    # Set the font family
    options.font_family = os.getenv('COVASIM_FONT_FAMILY', '')

    # Set the default font size -- if 0, use Matplotlib default
    options.dpi = int(os.getenv('COVASIM_DPI', 0))

    # Set whether or not to show figures -- default true
    options.show = int(os.getenv('COVASIM_SHOW', 1))

    # Set the figure backend -- only works globally
    options.backend = os.getenv('COVASIM_BACKEND', '')

    # Set default verbosity
    options.verbose = float(os.getenv('COVASIM_VERBOSE', 0.1))

    # Set default arithmetic precision -- use 32-bit by default for speed and memory efficiency
    options.precision = int(os.getenv('COVASIM_PRECISION', 32))

    # Specify whether to allow parallel Numba calculation -- about 20% faster, but the random number stream becomes nondeterministic
    options.numba_parallel = bool(int(os.getenv('COVASIM_NUMBA_PARALLEL', 0)))

    return options


# Actually set the options
options = set_default_options()

# Specify which keys require a reload
matplotlib_keys = ['font_size', 'font_family', 'dpi', 'backend']
numba_keys = ['precision', 'numba_parallel']


def set_option(key=None, value=None, set_global=True, **kwargs):
    '''
    Set a parameter or parameters.

    Args:
        key        (str):    the parameter to modify
        value      (varies): the value to specify
        set_global (bool):   if true (default), sets plotting options globally (rather than just for Covasim)
        kwargs     (dict):   if supplied, set multiple key-value pairs

    Options are:

        - font_size:      the font size used for the plots (default: 10)
        - font_family:    the font family/face used for the plots (default: Open Sans)
        - dpi:            the overall DPI for the figure (default: 100)
        - show:           whether to show figures (default true)
        - backend:        which Matplotlib backend to use (must be set globally)
        - verbose:        default verbosity for simulations to use (default: 1)
        - precision:      the arithmetic to use in calculations (default: 32; other option is 64)
        - numba_parallel: whether to parallelize Numba (default false; faster but gives non-reproducible results)

    **Examples**::

        cv.options.set('font_size', 18)
        cv.options.set(font_size=18, show=False, backend='agg', precision=64)
    '''
    if key is not None:
        kwargs = sc.mergedicts(kwargs, {key:value})
    reload_required = False
    default_options = set_default_options()
    for key,value in kwargs.items():
        if key not in options:
            keylist = [k for k in options.keys() if k != 'set'] # Set is not a key
            keys = '\n'.join(keylist)
            errormsg = f'Option "{key}" not recognized; options are:\n{keys}\n\nSee help(cv.options.set) for more information.'
            raise sc.KeyNotFoundError(errormsg)
        else:
            if value is None:
                value = default_options[key]
            options[key] = value
            if key in numba_keys:
                reload_required = True
            if key in matplotlib_keys and set_global:
                set_matplotlib_global(key, value)
    if reload_required:
        reload_numba()
    return


def set_matplotlib_global(key, value):
    ''' Set a global option for Matplotlib -- not for users '''
    import pylab as pl
    if   key == 'font_size':   pl.rc('font', size=value)
    elif key == 'font_family': pl.rc('font', family=value)
    elif key == 'dpi':         pl.rc('figure', dpi=value)
    elif key == 'backend':     pl.switch_backend(value)
    else: raise sc.KeyNotFoundError(f'Key {key} not found')
    return


def reload_numba():
    '''
    Apply changes to Numba functions -- reloading modules is necessary for
    changes to propogate. Not necessary if cv.options.set() is used.

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


# Add this here to be more accessible to the user
options.set = set_option