'''
Define options for Covasim, mostly plotting and Numba options. All options should
be set using set(), e.g.::

    cv.options.set(font_size=18)

To reset default options, use::

    cv.options.set('default')
'''

import os
import pylab as pl
import sciris as sc

__all__ = ['options']


def set_default_options():
    '''
    Set the default options for Covasim -- not to be called by the user, use
    ``cv.options.set('defaults')`` instead.
    '''

    # Options acts like a class, but is actually an objdict for simplicity
    optdesc = sc.objdict() # Help for the options
    options = sc.objdict() # The options

    optdesc.verbose = 'Set default level of verbosity (i.e. logging detail)'
    options.verbose = float(os.getenv('COVASIM_VERBOSE', 0.1))

    optdesc.show = 'Set whether or not to show figures (i.e. call pl.show() automatically)'
    options.show = int(os.getenv('COVASIM_SHOW', True))

    optdesc.close = 'Set whether or not to close figures (i.e. call pl.close() automatically)'
    options.close = int(os.getenv('COVASIM_CLOSE', False))

    optdesc.backend = 'Set the Matplotlib backend (use "agg" for non-interactive)'
    options.backend = os.getenv('COVASIM_BACKEND', pl.get_backend())

    optdesc.dpi = 'Set the default DPI -- the larger this is, the larger the figures will be'
    options.dpi = int(os.getenv('COVASIM_DPI', pl.rcParams['figure.dpi']))

    optdesc.font_size = 'Set the default font size'
    options.font_size = int(os.getenv('COVASIM_FONT_SIZE', pl.rcParams['font.size']))

    optdesc.font_family = 'Set the default font family (e.g., Arial)'
    options.font_family = os.getenv('COVASIM_FONT_FAMILY', pl.rcParams['font.family'])

    optdesc.precision = 'Set arithmetic precision for Numba -- 32-bit by default for efficiency'
    options.precision = int(os.getenv('COVASIM_PRECISION', 32))

    optdesc.numba_parallel = 'Set Numba multithreading -- about 20% faster, but simulations become nondeterministic'
    options.numba_parallel = bool(int(os.getenv('COVASIM_NUMBA_PARALLEL', 0)))

    return options, optdesc


# Actually set the options
options, optdesc = set_default_options()
orig_options = sc.dcp(options) # Make a copy for referring back to later

# Specify which keys require a reload
matplotlib_keys = ['font_size', 'font_family', 'dpi', 'backend']
numba_keys = ['precision', 'numba_parallel']


def set_option(key=None, value=None, set_global=True, **kwargs):
    '''
    Set a parameter or parameters. Use ``cv.options.set('defaults')`` to reset all
    values to default, or ``cv.options.set(dpi='default')`` to reset one parameter
    to default. See ``cv.options.help()`` for more information.

    Args:
        key        (str):    the parameter to modify, or 'defaults' to reset eerything to default values
        value      (varies): the value to specify; use None or 'default' to reset to default
        set_global (bool):   if true (default), sets plotting options globally (rather than just for Covasim)
        kwargs     (dict):   if supplied, set multiple key-value pairs

    Options are (see also ``cv.options.help()``):

        - verbose:        default verbosity for simulations to use
        - font_size:      the font size used for the plots
        - font_family:    the font family/face used for the plots
        - dpi:            the overall DPI for the figure
        - show:           whether to show figures
        - close:          whether to close the figures
        - backend:        which Matplotlib backend to use

        - precision:      the arithmetic to use in calculations
        - numba_parallel: whether to parallelize Numba

    **Examples**::

        cv.options.set('font_size', 18)
        cv.options.set(font_size=18, show=False, backend='agg', precision=64)
        cv.options.set('defaults') # Reset to default options
    '''

    if key is not None:
        kwargs = sc.mergedicts(kwargs, {key:value})
    reload_required = False

    # Reset to defaults
    if key in ['default', 'defaults']:
        kwargs = orig_options # Reset everything to default

    # Reset options
    for key,value in kwargs.items():
        if key not in options:
            keylist = orig_options.keys()
            keys = '\n'.join(keylist)
            errormsg = f'Option "{key}" not recognized; options are "defaults" or:\n{keys}\n\nSee help(cv.options.set) for more information.'
            raise sc.KeyNotFoundError(errormsg)
        else:
            if value in [None, 'default']:
                value = orig_options[key]
            options[key] = value
            if key in numba_keys:
                reload_required = True
            if key in matplotlib_keys and set_global:
                set_matplotlib_global(key, value)
    if reload_required:
        reload_numba()
    return


def get_help(output=False):
    '''
    Print information about options.

    Args:
        output (bool): whether to return a list of the options

    **Example**::

        cv.options.help()
    '''

    optdict = sc.objdict()
    for key in orig_options.keys():
        entry = sc.objdict()
        entry.key = key
        entry.current = options[key]
        entry.default = orig_options[key]
        entry.variable = f'COVASIM_{key.upper()}' # NB, hard-coded above!
        entry.desc = optdesc[key]
        optdict[key] = entry

    # Convert to a dataframe for nice printing
    print('Covasim global options ("Environment" = name of corresponding environment variable):')
    for key,entry in optdict.items():
        print(f'\n{key}')
        changestr = '' if entry.current == entry.default else ' (modified)'
        print(f'      Current: {entry.current}{changestr}')
        print(f'      Default: {entry.default}')
        print(f'  Environment: {entry.variable}')
        print(f'  Description: {entry.desc}')

    if output:
        return optdict
    else:
        return


def set_matplotlib_global(key, value):
    ''' Set a global option for Matplotlib -- not for users '''
    import pylab as pl
    if value: # Don't try to reset any of these to a None value
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


# Add these here to be more accessible to the user
options.set = set_option
options.help = get_help