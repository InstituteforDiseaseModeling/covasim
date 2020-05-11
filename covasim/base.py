'''
Base classes for Covasim.
'''

import numpy as np
import pandas as pd
import sciris as sc
import datetime as dt
from . import utils as cvu
from . import misc as cvm
from . import defaults as cvd
from . import plotting as cvplt

# Specify all externally visible classes this file defines
__all__ = ['ParsObj', 'Result', 'BaseSim', 'BasePeople', 'Person', 'FlexDict', 'Contacts', 'Layer', 'TransTree']


#%% Define simulation classes

class ParsObj(sc.prettyobj):
    '''
    A class based around performing operations on a self.pars dict.
    '''

    def __init__(self, pars):
        self.update_pars(pars, create=True)
        return

    def __getitem__(self, key):
        ''' Allow sim['par_name'] instead of sim.pars['par_name'] '''
        try:
            return self.pars[key]
        except:
            all_keys = '\n'.join(list(self.pars.keys()))
            errormsg = f'Key "{key}" not found; available keys:\n{all_keys}'
            raise sc.KeyNotFoundError(errormsg)

    def __setitem__(self, key, value):
        ''' Ditto '''
        if key in self.pars:
            self.pars[key] = value
        else:
            all_keys = '\n'.join(list(self.pars.keys()))
            errormsg = f'Key "{key}" not found; available keys:\n{all_keys}'
            raise sc.KeyNotFoundError(errormsg)
        return

    def update_pars(self, pars=None, create=False):
        '''
        Update internal dict with new pars.

        Args:
            pars (dict): the parameters to update (if None, do nothing)
            create (bool): if create is False, then raise a KeyNotFoundError if the key does not already exist
        '''
        if pars is not None:
            if not isinstance(pars, dict):
                raise TypeError(f'The pars object must be a dict; you supplied a {type(pars)}')
            if not hasattr(self, 'pars'):
                self.pars = pars
            if not create:
                available_keys = list(self.pars.keys())
                mismatches = [key for key in pars.keys() if key not in available_keys]
                if len(mismatches):
                    errormsg = f'Key(s) {mismatches} not found; available keys are {available_keys}'
                    raise sc.KeyNotFoundError(errormsg)
            self.pars.update(pars)
        return


class Result(object):
    '''
    Stores a single result -- by default, acts like an array.

    Args:
        name (str): name of this result, e.g. new_infections
        npts (int): if values is None, precreate it to be of this length
        scale (str): whether or not the value scales by population size; options are "dynamic", "static", or False
        color (str or array): default color for plotting (hex or RGB notation)

    **Example**::

        import covasim as cv
        r1 = cv.Result(name='test1', npts=10)
        r1[:5] = 20
        print(r1.values)
    '''

    def __init__(self, name=None, npts=None, scale='dynamic', color=None):
        self.name =  name  # Name of this result
        self.scale = scale # Whether or not to scale the result by the scale factor
        if color is None:
            color = '#000000'
        self.color = color # Default color
        if npts is None:
            npts = 0
        self.values = np.array(np.zeros(int(npts)), dtype=cvd.result_float)
        self.low    = None
        self.high   = None
        return

    def __repr__(self, *args, **kwargs):
        ''' Use pretty repr, like sc.prettyobj, but displaying full values '''
        output  = sc.prepr(self, skip=['values', 'low', 'high'])
        output += 'values:\n' + repr(self.values)
        if self.low is not None:
            output += '\nlow:\n' + repr(self.low)
        if self.high is not None:
            output += '\nhigh:\n' + repr(self.high)
        return output

    def __getitem__(self, *args, **kwargs):
        return self.values.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.values.__setitem__(*args, **kwargs)

    @property
    def npts(self):
        return len(self.values)


class BaseSim(ParsObj):
    '''
    The BaseSim class handles the running of the simulation: the number of people,
    number of time points, and the parameters of the simulation.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Initialize and set the parameters as attributes
        return

    def set_seed(self, seed=-1):
        '''
        Set the seed for the random number stream from the stored or supplied value

        Args:
            seed (None or int): if no argument, use current seed; if None, randomize; otherwise, use and store supplied seed

        Returns:
            None
        '''
        # Unless no seed is supplied, reset it
        if seed != -1:
            self['rand_seed'] = seed
        cvu.set_seed(self['rand_seed'])
        return

    @property
    def n(self):
        ''' Count the number of people -- if it fails, assume none '''
        try: # By default, the length of the people dict
            return len(self.people)
        except: # If it's None or missing
            return 0

    @property
    def npts(self):
        ''' Count the number of time points '''
        try:
            return int(self['n_days'] + 1)
        except:
            return 0

    @property
    def tvec(self):
        ''' Create a time vector '''
        try:
            return np.arange(self.npts)
        except:
            return np.array([])

    @property
    def datevec(self):
        '''
        Create a vector of dates

        Returns:
            Array of `datetime` instances containing the date associated with each
            simulation time step

        '''
        try:
            return self['start_day'] + self.tvec * dt.timedelta(days=1)
        except:
            return np.array([])


    def day(self, day, *args):
        '''
        Convert a string, date/datetime object, or int to a day (int).

        Args:
            day (str, date, int, or list): convert any of these objects to a day relative to the simulation's start day

        Returns:
            days (int or str): the day(s) in simulation time

        **Example**::

            sim.day('2020-04-05') # Returns 35
        '''
        # Do not process a day if it's not supplied
        if day is None:
            return None

        # Convert to list
        if sc.isstring(day) or sc.isnumber(day) or isinstance(day, (dt.date, dt.datetime)):
            day = sc.promotetolist(day) # Ensure it's iterable
        day.extend(args)

        days = []
        for d in day:
            if sc.isnumber(d):
                days.append(int(d)) # Just convert to an integer
            else:
                try:
                    if sc.isstring(d):
                        d = sc.readdate(d).date()
                    elif isinstance(d, dt.datetime):
                        d = d.date()
                    d_day = (d - self['start_day']).days
                    days.append(d_day)
                except Exception as E:
                    errormsg = f'Could not interpret "{d}" as a date: {str(E)}'
                    raise ValueError(errormsg)

        # Return an integer rather than a list if only one provided
        if len(days)==1:
            days = days[0]

        return days


    def date(self, ind, *args, dateformat=None, as_date=False):
        '''
        Convert one or more integer days of simulation time to a date/list of dates --
        by default returns a string, or returns a datetime Date object if as_date is True.

        Args:
            ind (int, list, or array): the day(s) in simulation time
            as_date (bool): whether to return as a datetime date instead of a string

        Returns:
            dates (str, Date, or list): the date(s) corresponding to the simulation day(s)

        **Examples**::

            sim.date(34) # Returns '2020-04-04'
            sim.date([34, 54]) # Returns ['2020-04-04', '2020-04-24']
            sim.date(34, 54, as_dt=True) # Returns [datetime.date(2020, 4, 4), datetime.date(2020, 4, 24)]
        '''

        # Handle inputs
        if sc.isnumber(ind): # If it's a number, convert it to a list
            ind = sc.promotetolist(ind)
        ind.extend(args)
        if dateformat is None:
            dateformat = '%Y-%m-%d'

        # Do the conversion
        dates = []
        for i in ind:
            date_obj = self['start_day'] + dt.timedelta(days=int(i))
            if as_date:
                dates.append(date_obj)
            else:
                dates.append(date_obj.strftime(dateformat))

        # Return a string rather than a list if only one provided
        if len(ind)==1:
            dates = dates[0]

        return dates


    def result_keys(self):
        ''' Get the actual results objects, not other things stored in sim.results '''
        keys = [key for key in self.results.keys() if isinstance(self.results[key], Result)]
        return keys


    def copy(self):
        ''' Returns a deep copy of the sim '''
        return sc.dcp(self)


    def export_results(self, for_json=True, filename=None, indent=2, *args, **kwargs):
        '''
        Convert results to dict -- see also to_json().

        The results written to Excel must have a regular table shape, whereas
        for the JSON output, arbitrary data shapes are supported.

        Args:
            for_json (bool): if False, only data associated with Result objects will be included in the converted output
            filename (str): filename to save to; if None, do not save
            indent (int): indent (int): if writing to file, how many indents to use per nested level
            args (list): passed to savejson()
            kwargs (dict): passed to savejson()

        Returns:
            resdict (dict): dictionary representation of the results

        '''
        resdict = {}
        resdict['t'] = self.results['t'] # Assume that there is a key for time

        if for_json:
            resdict['timeseries_keys'] = self.result_keys()
        for key,res in self.results.items():
            if isinstance(res, Result):
                resdict[key] = res.values
            elif for_json:
                if key == 'date':
                    resdict[key] = [str(d) for d in res] # Convert dates to strings
                else:
                    resdict[key] = res
        if filename is not None:
            sc.savejson(filename=filename, obj=resdict, indent=indent, *args, **kwargs)
        return resdict


    def export_pars(self, filename=None, indent=2, *args, **kwargs):
        '''
        Return parameters for JSON export -- see also to_json().

        This method is required so that interventions can specify
        their JSON-friendly representation.

        Args:
            filename (str): filename to save to; if None, do not save
            indent (int): indent (int): if writing to file, how many indents to use per nested level
            args (list): passed to savejson()
            kwargs (dict): passed to savejson()

        Returns:
            pardict (dict): a dictionary containing all the parameter values
        '''
        pardict = {}
        for key in self.pars.keys():
            if key == 'interventions':
                pardict[key] = [intervention.to_json() for intervention in self.pars[key]]
            elif key == 'start_day':
                pardict[key] = str(self.pars[key])
            else:
                pardict[key] = self.pars[key]
        if filename is not None:
            sc.savejson(filename=filename, obj=pardict, indent=indent, *args, **kwargs)
        return pardict


    def to_json(self, filename=None, keys=None, tostring=False, indent=2, verbose=False, *args, **kwargs):
        '''
        Export results as JSON.

        Args:
            filename (str): if None, return string; else, write to file
            keys (str or list): attributes to write to json (default: results, parameters, and summary)
            tostring (bool): if not writing to file, whether to write to string (alternative is sanitized dictionary)
            indent (int): if writing to file, how many indents to use per nested level
            verbose (bool): detail to print
            args (list): passed to savejson()
            kwargs (dict): passed to savejson()

        Returns:
            A unicode string containing a JSON representation of the results,
            or writes the JSON file to disk

        **Examples**::

            json = sim.to_json()
            sim.to_json('results.json')
            sim.to_json('summary.json', keys='summary')
        '''

        # Handle keys
        if keys is None:
            keys = ['results', 'pars', 'summary']
        keys = sc.promotetolist(keys)

        # Convert to JSON-compatible format
        d = {}
        for key in keys:
            if key == 'results':
                resdict = self.export_results(for_json=True)
                d['results'] = resdict
            elif key in ['pars', 'parameters']:
                pardict = self.export_pars()
                d['parameters'] = pardict
            elif key == 'summary':
                d['summary'] = dict(sc.dcp(self.summary))
            else:
                try:
                    d[key] = sc.sanitizejson(getattr(self, key))
                except Exception as E:
                    errormsg = f'Could not convert "{key}" to JSON: {str(E)}; continuing...'
                    print(errormsg)

        if filename is None:
            output = sc.jsonify(d, tostring=tostring, indent=indent, verbose=verbose, *args, **kwargs)
        else:
            output = sc.savejson(filename=filename, obj=d, indent=indent, *args, **kwargs)

        return output


    def to_excel(self, filename=None):
        '''
        Export results as XLSX

        Args:
            filename (str): if None, return string; else, write to file

        Returns:
            An sc.Spreadsheet with an Excel file, or writes the file to disk

        '''
        resdict = self.export_results(for_json=False)
        result_df = pd.DataFrame.from_dict(resdict)
        result_df.index = self.tvec
        result_df.index.name = 'Day'

        par_df = pd.DataFrame.from_dict(sc.flattendict(self.pars, sep='_'), orient='index', columns=['Value'])
        par_df.index.name = 'Parameter'

        spreadsheet = sc.Spreadsheet()
        spreadsheet.freshbytes()
        with pd.ExcelWriter(spreadsheet.bytes, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, sheet_name='Results')
            par_df.to_excel(writer, sheet_name='Parameters')
        spreadsheet.load()

        if filename is None:
            output = spreadsheet
        else:
            output = spreadsheet.save(filename)

        return output


    def shrink(self, skip_attrs=None, in_place=True):
        '''
        "Shrinks" the simulation by removing the people, and returns
        a copy of the "shrunken" simulation. Used to reduce the memory required
        for saved files.

        Args:
            skip_attrs (list): a list of attributes to skip in order to perform the shrinking; default "people"

        Returns:
            shrunken_sim (Sim): a Sim object with the listed attributes removed
        '''

        # By default, skip people (~90% of memory) and the popdict (which is usually empty anyway)
        if skip_attrs is None:
            skip_attrs = ['popdict', 'people']

        # Create the new object, and copy original dict, skipping the skipped attributes
        if in_place:
            for attr in skip_attrs:
                setattr(self, attr, None)
            return
        else:
            shrunken_sim = object.__new__(self.__class__)
            shrunken_sim.__dict__ = {k:(v if k not in skip_attrs else None) for k,v in self.__dict__.items()}
            return shrunken_sim


    def save(self, filename=None, keep_people=None, skip_attrs=None, **kwargs):
        '''
        Save to disk as a gzipped pickle.

        Args:
            filename (str or None): the name or path of the file to save to; if None, uses stored
            kwargs: passed to makefilepath()

        Returns:
            filename (str): the validated absolute path to the saved file

        **Example**::

            sim.save() # Saves to a .sim file with the date and time of creation by default
        '''

        # Set keep_people based on whether or not we're in the middle of a run
        if keep_people is None:
            if self.initialized and not self.results_ready:
                keep_people = True
            else:
                keep_people = False

        # Handle the filename
        if filename is None:
            filename = self.simfile
        filename = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename # Store the actual saved filename

        # Handle the shrinkage and save
        if skip_attrs or not keep_people:
            obj = self.shrink(skip_attrs=skip_attrs, in_place=False)
        else:
            obj = self
        sc.saveobj(filename=filename, obj=obj)

        return filename


    @staticmethod
    def load(filename, *args, **kwargs):
        '''
        Load from disk from a gzipped pickle.

        Args:
            filename (str): the name or path of the file to save to
            kwargs: passed to sc.loadobj()

        Returns:
            sim (Sim): the loaded simulation object

        **Example**::

            sim = cv.Sim.load('my-simulation.sim')
        '''
        sim = cvm.load(filename, *args, **kwargs)
        if not isinstance(sim, BaseSim):
            errormsg = f'Cannot load object of {type(sim)} as a Sim object'
            raise TypeError(errormsg)
        return sim


#%% Define people classes

class BasePeople(sc.prettyobj):
    '''
    A class to handle all the boilerplate for people -- note that everything
    interesting happens in the People class.
    '''

    def __init__(self, pars=None, pop_size=None, *args, **kwargs):

        # Handle pars and population size
        self.pars = pars
        if pop_size is None:
            if pars is not None:
                pop_size = pars['pop_size']
            else:
                pop_size = 0
        pop_size = int(pop_size)
        self.pop_size = pop_size
        if pars is not None:
            n_days = pars['n_days']
        self.n_days = n_days

        # Other initialization
        self.t = 0 # Keep current simulation time
        self._lock = False # Prevent further modification of keys
        self.meta = cvd.PeopleMeta() # Store list of keys and dtypes
        self.contacts = None
        self.init_contacts() # Initialize the contacts
        self.transtree = TransTree(pop_size=pop_size, n_days=n_days) # Initialize the transmission tree

        return


    def __getitem__(self, key):
        ''' Allow people['attr'] instead of getattr(people, 'attr') '''
        try:
            return self.__dict__[key]
        except:
            errormsg = f'Key "{key}" is not a valid attribute of people'
            raise AttributeError(errormsg)


    def __setitem__(self, key, value):
        ''' Ditto '''
        if self._lock and key not in self.__dict__:
            errormsg = f'Key "{key}" is not a valid attribute of people'
            raise AttributeError(errormsg)
        self.__dict__[key] = value
        return


    def __len__(self):
        ''' This is just a scalar, but validate() and resize() make sure it's right '''
        return self.pop_size


    def __iter__(self):
        ''' Define the iterator to just be the indices of the array '''
        return iter(range(len(self)))


    def __add__(self, people2):
        ''' Combine two people arrays '''
        newpeople = sc.dcp(self)
        for key in self.keys():
            newpeople.set(key, np.concatenate([newpeople[key], people2[key]]), die=False) # Allow size mismatch

        # Validate
        newpeople.pop_size += people2.pop_size
        newpeople.validate()

        # Reassign UIDs so they're unique
        newpeople.set('uid', np.arange(len(newpeople)))

        return newpeople


    def set(self, key, value, die=True):
        ''' Ensure sizes and dtypes match '''
        current = self[key]
        value = np.array(value, dtype=self._dtypes[key]) # Ensure it's the right type
        if die and len(value) != len(current):
            errormsg = f'Length of new array does not match current ({len(value)} vs. {len(current)})'
            raise IndexError(errormsg)
        self[key] = value
        return


    def get(self, key):
        ''' Convenience method -- key can be string or list of strings '''
        if isinstance(key, str):
            return self[key]
        elif isinstance(key, list):
            arr = np.zeros((len(self), len(key)))
            for k,ky in enumerate(key):
                arr[:,k] = self[ky]
            return arr


    def true(self, key):
        ''' Return indices matching the condition '''
        return self[key].nonzero()[0]


    def false(self, key):
        ''' Return indices not matching the condition '''
        return (~self[key]).nonzero()[0]


    def defined(self, key):
        ''' Return indices of people who are not-nan '''
        return (~np.isnan(self[key])).nonzero()[0]


    def not_defined(self, key):
        ''' Return indices of people who are nan '''
        return np.isnan(self[key]).nonzero()[0]


    def count(self, key):
        ''' Count the number of people for a given key '''
        return (self[key]>0).sum()


    def count_not(self, key):
        ''' Count the number of people who do not have a property for a given key '''
        return (self[key]==0).sum()


    def keys(self, which=None):
        ''' Returns the name of the states '''
        if which is None:
            return self.meta.all_states[:]
        else:
            return getattr(self.meta, which)[:]


    def layer_keys(self):
        ''' Get the available contact keys -- set by beta_layer rather than contacts since only the former is required '''
        try:
            keys = list(self.pars['beta_layer'].keys())
        except: # If not initialized
            keys = []
        return keys


    def index(self):
        ''' The indices of the array '''
        return np.arange(len(self))


    def validate(self, die=True, verbose=False):

        # Check that the keys match
        contact_layer_keys = set(self.contacts.keys())
        beta_layer_keys    = set(self.pars['beta_layer'].keys())
        if contact_layer_keys != beta_layer_keys:
            errormsg = f'Parameters layers {beta_layer_keys} are not consistent with contact layers {contact_layer_keys}'
            raise ValueError(errormsg)

        # Check that the length of each array is consistent
        expected_len = len(self)
        for key in self.keys():
            actual_len = len(self[key])
            if actual_len != expected_len:
                if die:
                    errormsg = f'Length of key "{key}" did not match population size ({actual_len} vs. {expected_len})'
                    raise IndexError(errormsg)
                else:
                    if verbose:
                        print(f'Resizing "{key}" from {actual_len} to {expected_len}')
                    self.resize(keys=key)
        return


    def resize(self, pop_size=None, keys=None):
        ''' Resize arrays if any mismatches are found '''
        if pop_size is None:
            pop_size = len(self)
        self.pop_size = pop_size
        if keys is None:
            keys = self.keys()
        keys = sc.promotetolist(keys)
        for key in keys:
            self[key].resize(pop_size, refcheck=False)
        return


    def to_df(self):
        ''' Convert to a Pandas dataframe '''
        df = pd.DataFrame.from_dict({key:self[key] for key in self.keys()})
        return df


    def to_arr(self):
        ''' Return as numpy array '''
        arr = np.empty((len(self), len(self.keys())), dtype=cvd.default_float)
        for k,key in enumerate(self.keys()):
            if key == 'uid':
                arr[:,k] = np.arange(len(self))
            else:
                arr[:,k] = self[key]
        return arr


    def person(self, ind):
        ''' Method to create person from the people '''
        p = Person()
        for key in self.meta.all_states:
            setattr(p, key, self[key][ind])
        return p


    def to_people(self):
        ''' Return all people as a list '''
        people = []
        for p in self:
            person = self.person(p)
            people.append(person)
        return people


    def from_people(self, people, resize=True):
        ''' Convert a list of people back into a People object '''

        # Handle population size
        pop_size = len(people)
        if resize:
            self.resize(pop_size=pop_size)

        # Iterate over people -- slow!
        for p,person in enumerate(people):
            for key in self.keys():
                self[key][p] = getattr(person, key)

        return


    def init_contacts(self, reset=False):
        ''' Initialize the contacts dataframe with the correct columns and data types '''

        # Create the contacts dictionary
        contacts = Contacts(layer_keys=self.layer_keys())

        if self.contacts is None or reset: # Reset all
            self.contacts = contacts
        else: # Only replace specified keys
            for key,layer in contacts.items():
                self.contacts[key] = layer
        return


    def add_contacts(self, contacts, lkey=None, beta=None):
        ''' Add new contacts to the array '''

        if lkey is None:
            lkey = self.layer_keys()[0]
        if lkey not in self.contacts:
            self.contacts[lkey] = Layer()

        # Validate the supplied contacts
        if isinstance(contacts, Contacts):
            new_contacts = contacts
        elif isinstance(contacts, Layer):
            new_contacts = {}
            new_contacts[lkey] = contacts
        elif sc.checktype(contacts, 'array'):
            new_contacts = {}
            new_contacts[lkey] = pd.DataFrame(data=contacts)
        elif isinstance(contacts, dict):
            new_contacts = {}
            new_contacts[lkey] = pd.DataFrame.from_dict(contacts)
        elif isinstance(contacts, list): # Assume it's a list of contacts by person, not an edgelist
            new_contacts = self.make_edgelist(contacts) # Assume contains key info
        else:
            errormsg = f'Cannot understand contacts of type {type(contacts)}; expecting dataframe, array, or dict'
            raise TypeError(errormsg)

        # Ensure the columns are right and add values if supplied
        for lkey, new_layer in new_contacts.items():
            n = len(new_layer['p1'])
            if 'beta' not in new_layer or len(new_layer['beta']) != n:
                if beta is None:
                    beta = 1.0
                beta = cvd.default_float(beta)
                new_layer['beta'] = np.ones(n, dtype=cvd.default_float)*beta

            # Actually include them, and update properties if supplied
            for col in self.contacts[lkey].keys():
                self.contacts[lkey][col] = np.concatenate([self.contacts[lkey][col], new_layer[col]])
            self.contacts[lkey].validate()

        return


    def make_edgelist(self, contacts):
        '''
        Parse a list of people with a list of contacts per person and turn it
        into an edge list.
        '''

        # Parse the list
        lkeys = self.layer_keys()
        new_contacts = Contacts(layer_keys=lkeys)
        for lkey in lkeys:
            new_contacts[lkey]['p1']    = [] # Person 1 of the contact pair
            new_contacts[lkey]['p2']    = [] # Person 2 of the contact pair

        try:
            for p,cdict in enumerate(contacts):
                for lkey,p_contacts in cdict.items():
                    n = len(p_contacts) # Number of contacts
                    new_contacts[lkey]['p1'].extend([p]*n) # e.g. [4, 4, 4, 4]
                    new_contacts[lkey]['p2'].extend(p_contacts) # e.g. [243, 4538, 7,19]
        except KeyError:
            lkeystr = ', '.join(lkeys)
            errormsg = f'Layer "{lkey}" could not be loaded since it was not among parameter keys "{lkeystr}". Please update manually or via sim.reset_layer_pars().'
            raise sc.KeyNotFoundError(errormsg)

        # Turn into a dataframe
        for lkey in lkeys:
            new_layer = Layer()
            for ckey,value in new_contacts[lkey].items():
                new_layer[ckey] = np.array(value, dtype=new_layer.meta[ckey])
            new_contacts[lkey] = new_layer

        return new_contacts


    @staticmethod
    def remove_duplicates(df):
        ''' Sort the dataframe and remove duplicates -- note, not extensively tested '''
        p1 = df[['p1', 'p2']].values.min(1) # Reassign p1 to be the lower-valued of the two contacts
        p2 = df[['p1', 'p2']].values.max(1) # Reassign p2 to be the higher-valued of the two contacts
        df['p1'] = p1
        df['p2'] = p2
        df.sort_values(['p1', 'p2'], inplace=True) # Sort by p1, then by p2
        df.drop_duplicates(['p1', 'p2'], inplace=True) # Remove duplicates
        df = df[df['p1'] != df['p2']] # Remove self connections
        df.reset_index(inplace=True, drop=True)
        return df


class Person(sc.prettyobj):
    '''
    Class for a single person. Note: this is largely deprecated since sim.people
    is now based on arrays rather than being a list of people.
    '''
    def __init__(self, pars=None, uid=None, age=-1, sex=-1, contacts=None):
        self.uid         = uid # This person's unique identifier
        self.age         = cvd.default_float(age) # Age of the person (in years)
        self.sex         = cvd.default_int(sex) # Female (0) or male (1)
        self.contacts    = contacts # Contacts
        self.infected = [] #: Record the UIDs of all people this person infected
        self.infected_by = None #: Store the UID of the person who caused the infection. If None but person is infected, then it was an externally seeded infection
        return


class FlexDict(dict):
    '''
    A dict that allows more flexible element access: in addition to obj['a'],
    also allow obj[0]. Lightweight implementation of the Sciris odict class.
    '''

    def __getitem__(self, key):
        ''' Lightweight odict -- allow indexing by number, with low performance '''
        try:
            return super().__getitem__(key)
        except KeyError as KE:
            try: # Assume it's an integer
                dictkey = self.keys()[key]
                return self[dictkey]
            except:
                raise sc.KeyNotFoundError(KE) # Raise the original error

    def keys(self):
        return list(super().keys())

    def values(self):
        return list(super().values())

    def items(self):
        return list(super().items())


class Contacts(FlexDict):
    '''
    A simple (for now) class for storing different contact layers.
    '''
    def __init__(self, layer_keys=None):
        if layer_keys is not None:
            for key in layer_keys:
                self[key] = Layer()
        return

    def __repr__(self):
        ''' Use slightly customized repr'''
        keys_str = ', '.join(self.keys())
        output = f'Contacts({keys_str})\n'
        for key in self.keys():
            output += f'\n"{key}": '
            output += self[key].__repr__() + '\n'
        return output


    def __len__(self):
        ''' The length of the contacts is the length of all the layers '''
        output = 0
        for key in self.keys():
            try:
                output += len(self[key])
            except:
                pass
        return output


class Layer(FlexDict):
    ''' A tiny class holding a single layer of contacts '''

    def __init__(self, **kwargs):
        self.meta = {
            'p1':    cvd.default_int, # Person 1
            'p2':    cvd.default_int,  # Person 2
            'beta':  cvd.default_float, # Default transmissibility for this contact type
        }
        self.basekey = 'p1' # Assign a base key for calculating lengths and performing other operations

        # Initialize the keys of the layers
        for key,dtype in self.meta.items():
            self[key] = np.empty((0,), dtype=dtype)

        # Set data, if provided
        for key,value in kwargs.items():
            self[key] = value

        return


    def __len__(self):
        try:
            return len(self[self.basekey])
        except:
            return 0


    def __repr__(self):
        ''' Convert to a dataframe for printing '''
        keys_str = ', '.join(self.keys())
        output = f'Layer({keys_str})\n'
        output += self.to_df().__repr__()
        return output


    def validate(self):
        ''' Check the integrity of the layer: right types, right lengths '''
        n = len(self[self.basekey])
        for key,dtype in self.meta.items():
            if dtype:
                assert self[key].dtype == dtype
            assert n == len(self[key])
        return


    def to_df(self):
        ''' Convert to dataframe '''
        df = pd.DataFrame.from_dict(self)
        return df


    def from_df(self, df):
        ''' Convert from dataframe '''
        for key in self.meta.keys():
            self[key] = df[key].to_numpy()
        return self



class TransTree(sc.prettyobj):
    '''
    A class for holding a transmission tree. Sources and targets are both lists
    of the same length as the population size. Sources has one entry for people who
    have been infected, zero for those who haven't. Targets is a list of lists,
    with the length of the list being the number of people that person infected.

    Args:
        pop_size (int): the number of people in the population
    '''

    def __init__(self, pop_size, n_days):
        self.linelist = [None]*pop_size
        self.targets  = [[] for p in range(len(self))] # Make a list of empty lists
        self.detailed = None
        self.pop_size = pop_size
        self.n_days   = n_days
        return


    def __len__(self):
        '''
        The length of the transmission tree is the length of the line list,
        which should equal the population size (non-infected people are None
        in the line list).
        '''
        try:
            return len(self.linelist)
        except:
            return 0


    def make_targets(self, reset=False):
        '''
        Convert sources into targets -- same information, just grouped differently.
        Usually done inside sim:step(), here just for completeness.
        '''
        if self.targets is None or reset:
            self.targets = [[] for p in range(len(self))] # Make a list of empty lists
            for transdict in self.linelist:
                if transdict is not None:
                    source = transdict['source']
                    if source is not None: # e.g., from an importation
                        self.targets[source].append(transdict)
            return


    def make_detailed(self, people, reset=False):
        ''' Construct a detailed transmission tree, with additional information for each person '''
        if self.detailed is None or reset:

            # Reset to look like the line list, but with more detail
            self.detailed = [None]*len(self)

            for transdict in self.linelist:

                if transdict is not None:

                    # Pull out key quantities
                    ddict  = sc.objdict(sc.dcp(transdict)) # For "detailed dictionary"
                    source = ddict.source
                    target = ddict.target
                    ddict.s = sc.objdict() # Source
                    ddict.t = sc.objdict() # Target

                    # If the source is available (e.g. not a seed infection), loop over both it and the target
                    if source is not None:
                        stdict = {'s':source, 't':target}
                    else:
                        stdict = {'t':target}

                    # Pull out each of the attributes relevant to transmission
                    attrs = ['age', 'date_symptomatic', 'date_tested', 'date_diagnosed', 'date_quarantined', 'date_severe', 'date_critical', 'date_known_contact']
                    for st,stind in stdict.items():
                        for attr in attrs:
                            ddict[st][attr] = people[attr][stind]
                    if source is not None:
                        for attr in attrs:
                            if attr.startswith('date_'):
                                is_attr = attr.replace('date_', 'is_') # Convert date to a boolean, e.g. date_diagnosed -> is_diagnosed
                                ddict.s[is_attr] = ddict.s[attr] <= ddict['date'] # These don't make sense for people just infected (targets), only sources

                        ddict.s.is_asymp   = np.isnan(people.date_symptomatic[source])
                        ddict.s.is_presymp = ~ddict.s.is_asymp and ~ddict.s.is_symptomatic # Not asymptomatic and not currently symptomatic
                    ddict.t['is_quarantined'] = ddict.t['date_quarantined'] <= ddict['date'] # This is the only target date that it makes sense to define since it can happen before infection

                    self.detailed[target] = ddict

        return


    def plot(self, *args, **kwargs):
        ''' Plot the transmission tree '''
        return cvplt.plot_transtree(self, *args, **kwargs)


    def animate(self, *args, **kwargs):
        '''
        Animate the transmission tree.

        Args:
            animate    (bool):  whether to animate the plot (otherwise, show when finished)
            verbose    (bool):  print out progress of each frame
            markersize (int):   size of the markers
            sus_color  (list):  color for susceptibles
            fig_args   (dict):  arguments passed to pl.figure()
            axis_args  (dict):  arguments passed to pl.subplots_adjust()
            plot_args  (dict):  arguments passed to pl.plot()
            delay      (float): delay between frames in seconds
            font_size  (int):   size of the font
            colors     (list):  color of each person
            cmap       (str):   colormap for each person (if colors is not supplied)

        Returns:
            fig: the figure object
        '''
        return cvplt.animate_transtree(self, *args, **kwargs)