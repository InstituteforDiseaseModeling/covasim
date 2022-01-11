'''
Define options for Covasim, mostly plotting and Numba options. All options should
be set using set() or directly, e.g.::

    cv.options(font_size=18)

To reset default options, use::

    cv.options('default')

Note: "options" is used to refer to the choices available (e.g., DPI), while "settings"
is used to refer to the choices made (e.g., DPI=150).
'''

import os
import pylab as pl
import sciris as sc

# Only the class instance is public
__all__ = ['options']


#%% General settings

# Specify which keys require a reload
matplotlib_keys = ['backend', 'style', 'dpi', 'font_size', 'font_family']
numba_keys      = ['precision', 'numba_parallel', 'numba_cache']

# Define simple plotting options -- similar to Matplotlib default
rc_simple = {
    'figure.facecolor': 'white',
    'axes.spines.right': False,
    'axes.spines.top': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Muli'] + pl.rcParams['font.sans-serif'],
    'legend.frameon': False,
}

# Define default plotting options -- based on Seaborn
rc_covasim = sc.mergedicts(rc_simple, {
    'axes.facecolor': 'efefff',
    'axes.grid': True,
    'grid.color': 'white',
    'grid.linewidth': 1,
    'font.serif': ['Rosario', 'Garamond', 'Garamond MT'] + pl.rcParams['font.serif'],
})


#%% Define the options class

class Options(sc.objdict):
    '''
    Set options for Covasim. Use ``cv.options.set('defaults')`` to reset all
    values to default, or ``cv.options.set(dpi='default')`` to reset one parameter
    to default. See ``cv.options.help()`` for more information.

    Args:
        key    (str):    the parameter to modify, or 'defaults' to reset everything to default values
        value  (varies): the value to specify; use None or 'default' to reset to default
        kwargs (dict):   if supplied, set multiple key-value pairs

    Options are (see also ``cv.options.help()``):

        - verbose:        default verbosity for simulations to use
        - style:          the plotting style to use
        - font_size:      the font size used for the plots
        - font_family:    the font family/face used for the plots
        - dpi:            the overall DPI for the figure
        - show:           whether to show figures
        - close:          whether to close the figures
        - backend:        which Matplotlib backend to use
        - interactive:    convenience method to set show, close, and backend
        - precision:      the arithmetic to use in calculations
        - numba_parallel: whether to parallelize Numba functions
        - numba_cache:    whether to cache (precompile) Numba functions

    **Examples**::

        cv.options.set('font_size', 18) # Larger font
        cv.options.set(font_size=18, show=False, backend='agg', precision=64) # Larger font, non-interactive plots, higher precision
        cv.options.set(interactive=False) # Turn off interactive plots
        cv.options.set('defaults') # Reset to default options
        cv.options.set('jupyter') # Defaults for Jupyter

    | New in version 3.1.1: Jupyter defaults
    | New in version 3.1.2: Updated plotting styles; refactored options as a class
    '''

    def __init__(self):
        super().__init__()
        optdesc, options = self.get_orig_options() # Get the options
        self.update(options) # Update this object with them
        self.setattribute('optdesc', optdesc) # Set the description as an attribute, not a dict entry
        self.setattribute('orig_options', sc.dcp(options)) # Copy the default options
        return


    def __call__(self, *args, **kwargs):
        '''Allow ``cv.options(dpi=150)`` instead of ``cv.options.set(dpi=150)`` '''
        return self.set(*args, **kwargs)


    def to_dict(self):
        ''' Pull out only the settings from the options object '''
        return {k:v for k,v in self.items()}


    def __repr__(self):
        ''' Brief representation '''
        output = sc.objectid(self)
        output += 'Covasim options (see also cv.options.disp()):\n'
        output += sc.pp(self.to_dict(), output=True)
        return output


    def disp(self):
        ''' Detailed representation '''
        output = 'Covasim options (see also cv.options.help()):\n'
        keylen = 14 # Maximum key length  -- "numba_parallel"
        for k,v in self.items():
            keystr = sc.colorize(f'  {k:>{keylen}s}: ', fg='cyan', output=True)
            reprstr = sc.pp(v, output=True)
            reprstr = sc.indent(n=keylen+4, text=reprstr, width=None)
            output += f'{keystr}{reprstr}'
        print(output)
        return


    @staticmethod
    def get_orig_options():
        '''
        Set the default options for Covasim -- not to be called by the user, use
        ``cv.options.set('defaults')`` instead.
        '''

        # Options acts like a class, but is actually an objdict for simplicity
        optdesc = sc.objdict() # Help for the options
        options = sc.objdict() # The options

        optdesc.verbose = 'Set default level of verbosity (i.e. logging detail)'
        options.verbose = float(os.getenv('COVASIM_VERBOSE', 0.1))

        optdesc.sep = 'Set thousands seperator for text output'
        options.sep = str(os.getenv('COVASIM_SEP', ','))

        optdesc.show = 'Set whether or not to show figures (i.e. call pl.show() automatically)'
        options.show = int(os.getenv('COVASIM_SHOW', True))

        optdesc.close = 'Set whether or not to close figures (i.e. call pl.close() automatically)'
        options.close = int(os.getenv('COVASIM_CLOSE', False))

        optdesc.backend = 'Set the Matplotlib backend (use "agg" for non-interactive)'
        options.backend = os.getenv('COVASIM_BACKEND', pl.get_backend())

        optdesc.interactive = 'Convenience method to set figure backend, showing, and closing behavior'
        options.interactive = os.getenv('COVASIM_INTERACTIVE', True)

        optdesc.style = 'Set the default plotting style -- options are "covasim" and "simple" plus those in pl.style.available; see also options.rc'
        options.style = os.getenv('COVASIM_STYLE', 'covasim')

        optdesc.rc = 'Matplotlib rc (run control) parameters used during plotting'
        options.rc = sc.dcp(rc_covasim)

        optdesc.dpi = 'Set the default DPI -- the larger this is, the larger the figures will be'
        options.dpi = int(os.getenv('COVASIM_DPI', pl.rcParams['figure.dpi']))

        optdesc.font = 'Set the default font family (e.g., sans-serif or Arial)'
        options.font = os.getenv('COVASIM_FONT', pl.rcParams['font.family'])

        optdesc.fontsize = 'Set the default font size'
        options.fontsize = int(os.getenv('COVASIM_FONT_SIZE', pl.rcParams['font.size']))

        optdesc.precision = 'Set arithmetic precision for Numba -- 32-bit by default for efficiency'
        options.precision = int(os.getenv('COVASIM_PRECISION', 32))

        optdesc.numba_parallel = 'Set Numba multithreading -- none, safe, full; full multithreading is ~20% faster, but results become nondeterministic'
        options.numba_parallel = str(os.getenv('COVASIM_NUMBA_PARALLEL', 'none'))

        optdesc.numba_cache = 'Set Numba caching -- saves on compilation time; disabling is not recommended'
        options.numba_cache = bool(int(os.getenv('COVASIM_NUMBA_CACHE', 1)))

        return optdesc, options


    def set(self, key=None, value=None, **kwargs):

        reload_required = False

        # Reset to defaults
        if key in ['default', 'defaults']:
            kwargs = self.orig_options # Reset everything to default

        # Handle Jupyter
        elif sc.isstring(key) and 'jupyter' in key.lower():
            jupyter_kwargs = dict(
                dpi = 100,
                show = False,
                close = True,
            )
            kwargs = sc.mergedicts(jupyter_kwargs, kwargs)
            try: # This makes plots much nicer, but isn't available on all systems
                if not os.environ.get('SPHINX_BUILD'): # Custom check implemented in conf.py to skip this if we're inside Sphinx
                    import matplotlib_inline
                    matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
            except:
                pass

        # Handle other keys
        elif key is not None:
            kwargs = sc.mergedicts(kwargs, {key:value})

        # Handle interactivity
        if 'interactive' in kwargs.keys():
            interactive = kwargs['interactive']
            if interactive in [None, 'default']:
                interactive = self.orig_options['interactive']
            if interactive:
                kwargs['show'] = True
                kwargs['close'] = False
                kwargs['backend'] = self.orig_options['backend']
            else:
                kwargs['show'] = False
                kwargs['backend'] = 'agg'

        # Reset options
        for key,value in kwargs.items():
            if key not in self:
                keylist = self.orig_options.keys()
                keys = '\n'.join(keylist)
                errormsg = f'Option "{key}" not recognized; options are "defaults" or:\n{keys}\n\nSee help(cv.options.set) for more information.'
                raise sc.KeyNotFoundError(errormsg)
            else:
                if value in [None, 'default']:
                    value = self.orig_options[key]
                self[key] = value
                if key in numba_keys:
                    reload_required = True
                if key in matplotlib_keys:
                    set_matplotlib_global(key, value)

        if reload_required:
            reload_numba()
        return


    def get_default(self, key):
        ''' Helper function to get the original default options '''
        return self.orig_options[key]


    def changed(self, key):
        ''' Check if current setting has been changed from default '''
        if key in self.orig_options:
            return self[key] != self.orig_options[key]
        else:
            return None


    def help(self, output=False):
        '''
        Print information about options.

        Args:
            output (bool): whether to return a list of the options

        **Example**::

            cv.options.help()
        '''
        n = 15 # Size of indent
        optdict = sc.objdict()
        for key in self.orig_options.keys():
            entry = sc.objdict()
            entry.key = key
            entry.current = sc.indent(n=n, width=None, text=sc.pp(self[key], output=True)).rstrip()
            entry.default = sc.indent(n=n, width=None, text=sc.pp(self.orig_options[key], output=True)).rstrip()
            if not key.startswith('rc'):
                entry.variable = f'COVASIM_{key.upper()}' # NB, hard-coded above!
            else:
                entry.variable = 'No environment variable'
            entry.desc = sc.indent(n=n, text=self.optdesc[key])
            optdict[key] = entry

        # Convert to a dataframe for nice printing
        print('Covasim global options ("Environment" = name of corresponding environment variable):')
        for k, key,entry in optdict.enumitems():
            sc.heading(f'{k}. {key}', spaces=0, spacesafter=0)
            changestr = '' if entry.current == entry.default else ' (modified)'
            print(f'          Key: {key}')
            print(f'      Current: {entry.current}{changestr}')
            print(f'      Default: {entry.default}')
            print(f'  Environment: {entry.variable}')
            print(f'  Description: {entry.desc}')

        sc.heading('Methods:', spacesafter=0)
        print('''
    cv.options(key=value) -- set key to value
    cv.options[key] -- get or set key
    cv.options.set() -- set option(s)
    cv.options.get_default() -- get default setting(s)
    cv.options.load() -- load settings from file
    cv.options.save() -- save settings to file
    cv.options.to_dict() -- convert to dictionary
''')

        if output:
            return optdict
        else:
            return


    def load(self, filename, verbose=True, **kwargs):
        '''
        Save current settings as a JSON file.

        Args:
            filename (str): file to load
            kwargs (dict): passed to ``sc.loadjson()``
        '''
        json = sc.loadjson(filename=filename, **kwargs)
        current = self.to_dict()
        new = {k:v for k,v in json.items() if v != current[k]} # Don't reset keys that haven't changed
        self.set(**new)
        if verbose: print(f'Settings loaded from {filename}')
        return


    def save(self, filename, verbose=True, **kwargs):
        '''
        Save current settings as a JSON file.

        Args:
            filename (str): file to save to
            kwargs (dict): passed to ``sc.savejson()``
        '''
        json = self.to_dict()
        output = sc.savejson(filename=filename, obj=json, **kwargs)
        if verbose: print(f'Settings saved to {filename}')
        return output


    def style(self, use=False, **kwargs):
        '''
        Combine all Matplotlib style information, and either apply it directly
        or create a style context.

        Args:
            use (bool): whether to set as the global style; else, treat as context for use with "with" (default)
        '''
        # Handle inputs
        rc = sc.dcp(self.rc) # Make a local copy of the currently used settings

        # Handle style, overwiting existing
        style = kwargs.pop('style')
        stylestr = str(style).lower()
        if stylestr in ['none', 'default', 'covasim', 'house']:
            rc = sc.dcp(rc_covasim)
        elif stylestr in ['simple', 'covasim_simple']:
            rc = sc.dcp(rc_simple)
        elif style in pl.style.library:
            rc = pl.style.library[style]
        else:
            errormsg = f'Could not apply style "{style}": please use "covasim", "simple", or one of the styles from pl.styles.available'
            raise ValueError(errormsg)


        def pop_keywords(sourcekeys, rckey):
            ''' Helper function to handle input arguments '''
            sourcekeys = sc.tolist(sourcekeys)
            key = sourcekeys[0] # Main key
            value = None
            changed = self.changed(key)
            if changed:
                value = self[key]
            for k in sourcekeys:
                value = kwargs.pop(k, value)
            if value is not None:
                rc[rckey] = value
            return

        # Handle special cases
        pop_keywords('dpi', rckey='figure.dpi')
        pop_keywords(['font', 'fontfamily', 'font_family'], rckey='font.family')
        pop_keywords(['fontsize', 'font_size'], rckey='font.size')
        pop_keywords('grid', rckey='axes.grid')
        pop_keywords('facecolor', rckey='axes.facecolor')

        # Handle other keywords
        for key,value in kwargs.items():
            if key not in pl.rcParams:
                errormsg = f'Key "{key}" does not match any value in Covasim options or pl.rcParams'
                raise sc.KeyNotFoundError(errormsg)
            elif value is not None:
                rc[key] = value

        # Tidy up
        if use:
            return pl.style.use(rc)
        else:
            return pl.style.context(rc)


def set_matplotlib_global(key, value, available_fonts=None):
    ''' Set a global option for Matplotlib -- not for users '''
    if value: # Don't try to reset any of these to a None value
        if   key == 'font_size':   pl.rcParams['font.size']   = value
        elif key == 'dpi':         pl.rcParams['figure.dpi']  = value
        elif key == 'backend':     pl.switch_backend(value)
        elif key == 'font_family':
            if available_fonts is None or value in available_fonts: # If available fonts are supplied, don't set to an invalid value
                pl.rcParams['font.family'] = value
        elif key == 'style':
            if value is None or value.lower() == 'covasim':
                pl.style.use('default')
            elif value in pl.style.available:
                pl.style.use(value)
            else:
                errormsg = f'Style "{value}"; not found; options are "covasim" (default) plus:\n{sc.newlinejoin(pl.style.available)}'
                raise ValueError(errormsg)
        else: raise sc.KeyNotFoundError(f'Key {key} not found')
    return


def reload_numba():
    '''
    Apply changes to Numba functions -- reloading modules is necessary for
    changes to propagate. Not necessary to call directly if cv.options.set() is used.

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
    print("Reload complete. Note: for some options to take effect, you may also need to delete Covasim's __pycache__ folder.")
    return


def load_fonts(folder=None):
    '''
    Load custom fonts for plotting -- alias to ``sc.fonts()``
    '''
    if folder is None:
        folder = str(sc.thisdir(__file__, aspath=True) / 'data' / 'assets')
    sc.fonts(add=folder)
    return


# Create the options on module load, and load the fonts
options = Options()
load_fonts()
