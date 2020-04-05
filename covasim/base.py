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

    def update_pars(self, pars=None, create=False):
        '''
        Update internal dict with new pars.

        Args:
            pars (dict): the parameters to update (if None, do nothing)
            create (bool): if create is False, then raise a KeyError if the key does not already exist
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
                    raise KeyError(errormsg)
            self.pars.update(pars)
        return


class Result(object):
    '''
    Stores a single result -- by default, acts like an array.

    Args:
        name (str): name of this result, e.g. new_infections
        values (array): array of values corresponding to this result
        npts (int): if values is None, precreate it to be of this length
        scale (bool): whether or not the value scales by population size
        color (str or array): default color for plotting (hex or RGB notation)

    Example:
        import covasim as cv
        r1 = cv.Result(name='test1', npts=10)
        r1[:5] = 20
        print(r2.values)
        r2 = cv.Result(name='test2', values=range(10))
        print(r2)
    '''

    def __init__(self, name=None, values=None, npts=None, scale=True, color=None):
        self.name =  name  # Name of this result
        self.scale = scale # Whether or not to scale the result by the scale factor
        if color is None:
            color = '#000000'
        self.color = color # Default color
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
        ''' Count the number of people -- if it fails, assume none '''
        try: # By default, the length of the people dict
            output = len(self.people)
        except: # If it's None or missing
            output = 0
        return output

    @property
    def npts(self):
        ''' Count the number of time points '''
        return int(self['n_days'] + 1)

    @property
    def tvec(self):
        ''' Create a time vector '''
        return np.arange(self.npts)


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
        """
        Convert results to dict

        The results written to Excel must have a regular table shape, whereas
        for the JSON output, arbitrary data shapes are supported.

        Args:
            for_json: If False, only data associated with Result objects will be included in the converted output

        Returns: Dictionary representation of the results

        """
        resdict = {}
        resdict['t'] = self.results['t'] # Assume that there is a key for time

        if for_json:
            resdict['timeseries_keys'] = self.reskeys
        for key,res in self.results.items():
            if isinstance(res, Result):
                resdict[key] = res.values
            elif for_json:
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


    def shrink(self, skip_attrs=None):
        '''
        "Shrinks" the simulation by removing the people and UIDs, and returns
        a copy of the "shrunken" simulation. Used to reduce the memory required
        for saved files.

        Args:
            skip_attrs (list): a list of attributes to skip in order to perform the shrinking; default "people", "popdict", and "uids"

        Returns:
            shrunken_sim (Sim): a Sim object with the listed attributes removed
        '''

        # By default, skip people (~90%) and uids (~9%)
        if skip_attrs is None:
            skip_attrs = ['popdict', 'uids', 'people']

        # Create the new object, and copy original dict, skipping the skipped attributes
        shrunken_sim = object.__new__(self.__class__)
        shrunken_sim.__dict__ = {k:(v if k not in skip_attrs else None) for k,v in self.__dict__.items()}

        return shrunken_sim


    def save(self, filename=None, keep_people=False, skip_attrs=None, **kwargs):
        '''
        Save to disk as a gzipped pickle.

        Args:
            filename (str or None): the name or path of the file to save to; if None, uses stored
            kwargs: passed to makefilepath()

        Returns:
            filename (str): the validated absolute path to the saved file

        Example:
            sim.save() # Saves to a .sim file with the date and time of creation by default
        '''
        if filename is None:
            filename = self.filename
        filename = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename # Store the actual saved filename
        if skip_attrs or not keep_people:
            obj = self.shrink(skip_attrs=skip_attrs)
        else:
            obj = self
        sc.saveobj(filename=filename, obj=obj)
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