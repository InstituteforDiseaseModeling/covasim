'''
Base classes for Covasim.
'''

#%% Imports
import datetime as dt
import numpy as np # Needed for a few things not provided by pl
import sciris as sc
import pandas as pd
from . import utils as cov_ut

# Specify all externally visible functions this file defines
__all__ = ['ParsObj', 'Result', 'BaseSim']



#%% Define classes
class ParsObj(sc.prettyobj):
    '''
    A class based around performing operations on a self.pars dict.
    '''

    def __init__(self, pars):
        self.update_pars(pars, create=True)
        return

    def __getitem__(self, key):
        ''' Allow sim['par_name'] instead of sim.pars['par_name'] '''
        return self.pars[key]

    def __setitem__(self, key, value):
        ''' Ditto '''
        if key in self.pars:
            self.pars[key] = value
        else:
            suggestion = sc.suggest(key, self.pars.keys())
            if suggestion:
                errormsg = f'Key {key} not found; did you mean "{suggestion}"?'
            else:
                all_keys = '\n'.join(list(self.pars.keys()))
                errormsg = f'Key {key} not found; available keys:\n{all_keys}'
            raise KeyError(errormsg)
        return

    def update_pars(self, pars, create=False):
        '''
        Update internal dict with new pars. If create is False, then raise a KeyError
        if the key does not already exist.
        '''
        if not isinstance(pars, dict):
            raise TypeError(f'The pars object must be a dict; you supplied a {type(pars)}')
        if not hasattr(self, 'pars'):
            self.pars = pars
        elif pars is not None:
            if not create:
                available_keys = list(self.pars.keys())
                mismatches = [key for key in pars.keys() if key not in available_keys]
                if len(mismatches):
                    errormsg = f'Key(s) {mismatches} not found; available keys are {available_keys}'
                    raise KeyError(errormsg)
            self.pars.update(pars)
        return


class Result(object):
    '''
    Stores a single result -- by default, acts like an array.

    Example:
        import covasim as cova
        r1 = cova.Result(name='test1', npts=10)
        r1[:5] = 20
        print(r2.values)
        r2 = cova.Result(name='test2', values=range(10))
        print(r2)
    '''

    def __init__(self, name=None, values=None, npts=None, scale=True, ispercentage=False):
        self.name = name  # Name of this result
        self.ispercentage = ispercentage  # Whether or not the result is a percentage
        self.scale = scale  # Whether or not to scale the result by the scale factor
        if values is None:
            if npts is not None:
                values = np.zeros(int(npts)) # If length is known, use zeros
            else:
                values = [] # Otherwise, empty
        self.values = np.array(values, dtype=float) # Ensure it's an array
        return

    def __repr__(self, *args, **kwargs):
        ''' Use pretty repr, like sc.prettyobj, but displaying full values '''
        output  = sc.prepr(self, skip='values')
        output += 'values:\n' + repr(self.values)
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

    def set_seed(self, seed=-1) -> None:
        """
        Set the seed for the random number stream from the stored or supplied value

        Args:
            seed (None or int): if no argument, use current seed; if None, randomize; otherwise, use and store supplied seed

        Returns:
            None
        """
        # Unless no seed is supplied, reset it
        if seed != -1:
            self['seed'] = seed
        cov_ut.set_seed(self['seed'])
        return

    @property
    def n(self):
        ''' Count the number of people '''
        return len(self.people)

    @property
    def npts(self):
        ''' Count the number of time points '''
        return int(self['n_days'] + 1)

    @property
    def tvec(self):
        ''' Create a time vector '''
        return np.arange(self['n_days'] + 1)


    def inds2dates(self, inds, dateformat=None):
        ''' Convert a set of indices to a set of dates '''

        if sc.isnumber(inds): # If it's a number, convert it to a list
            inds = sc.promotetolist(inds)

        if dateformat is None:
            dateformat = '%b-%d'

        dates = []
        for ind in inds:
            tmp = self['start_day'] + dt.timedelta(days=int(ind))
            dates.append(tmp.strftime(dateformat))
        return dates


    def get_person(self, ind):
        ''' Return a person based on their index '''
        return self.people[self.uids[int(ind)]]


    def _make_resdict(self, for_json: bool = True) -> dict:
        ''' Pre-convert the results structure to a friendier output'''
        resdict = {}
        if for_json:
            resdict['timeseries_keys'] = self.reskeys
        for key,res in self.results.items():
            if isinstance(res, Result):
                res = res.values
            if for_json or sc.isiterable(res) and len(res)==self.npts:
                resdict[key] = res
        return resdict

    def _make_pardict(self) -> dict:
        """
        Return parameters for JSON export

        This method is required so that interventions can specify
        their JSON-friendly representation

        Returns:

        """
        pardict = self.pars
        pardict['interventions'] = [intervention.to_json() for intervention in pardict['interventions']]
        return pardict

    def to_json(self, filename=None, tostring=True, indent=2, *args, **kwargs):
        """
        Export results as JSON.

        Args:
            filename (str): if None, return string; else, write to file

        Returns:
            A unicode string containing a JSON representation of the results,
            or writes the JSON file to disk

        """
        resdict = self._make_resdict()
        pardict = self._make_pardict()
        d = {'results': resdict, 'parameters': pardict}
        if filename is None:
            output = sc.jsonify(d, tostring=tostring, indent=indent, *args, **kwargs)
        else:
            output = sc.savejson(filename=filename, obj=d, *args, **kwargs)

        return output


    def to_xlsx(self, filename=None):
        """
        Export results as XLSX

        Args:
            filename (str): if None, return string; else, write to file

        Returns:
            An sc.Spreadsheet with an Excel file, or writes the file to disk

        """
        resdict = self._make_resdict(for_json=False)
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


    def save(self, filename=None, **kwargs):
        '''
        Save to disk as a gzipped pickle.

        Args:
            filename (str or None): the name or path of the file to save to; if None, uses stored
            keywords: passed to makefilepath()

        Returns:
            filename (str): the validated absolute path to the saved file

        Example:
            sim.save() # Saves to a .sim file with the date and time of creation by default

        '''
        if filename is None:
            filename = self.filename
        filename = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename # Store the actual saved filename
        sc.saveobj(filename=filename, obj=self)
        return filename


    @staticmethod
    def load(filename, **kwargs):
        '''
        Load from disk from a gzipped pickle.

        Args:
            filename (str): the name or path of the file to save to
            keywords: passed to makefilepath()

        Returns:
            sim (Sim): the loaded simulation object

        Example:
            sim = cv.Sim.load('my-simulation.sim')
        '''
        filename = sc.makefilepath(filename=filename, **kwargs)
        sim = sc.loadobj(filename=filename)
        return sim