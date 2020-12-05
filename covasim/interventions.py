'''
Specify the core interventions available in Covasim. Other interventions can be
defined by the user by inheriting from these classes.
'''

import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
import inspect
import datetime as dt
from . import utils as cvu
from . import defaults as cvd
from . import base as cvb
from . import parameters as cvpar
from collections import defaultdict


#%% Generic intervention classes

__all__ = ['InterventionDict', 'Intervention', 'dynamic_pars', 'sequence']


def find_day(arr, t=None, which='first'):
    '''
    Helper function to find if the current simulation time matches any day in the
    intervention. Although usually never more than one index is returned, it is
    returned as a list for the sake of easy iteration.

    Args:
        arr (list): list of days in the intervention, or else a boolean array
        t (int): current simulation time (can be None if a boolean array is used)
        which (str): what to return: 'first', 'last', or 'all' indices

    Returns:
        inds (list): list of matching days; length zero or one unless which is 'all'
    '''
    all_inds = sc.findinds(arr=arr, val=t)
    if len(all_inds) == 0 or which == 'all':
        inds = all_inds
    elif which == 'first':
        inds = [all_inds[0]]
    elif which == 'last':
        inds = [all_inds[-1]]
    else:
        errormsg = f'Argument "which" must be "first", "last", or "all", not "{which}"'
        raise ValueError(errormsg)
    return inds


def InterventionDict(which, pars):
    '''
    Generate an intervention from a dictionary. Although a function, it acts
    like a class, since it returns a class instance.

    **Example**::

        interv = cv.InterventionDict(which='change_beta', pars={'days': 30, 'changes': 0.5, 'layers': None})
    '''
    mapping = dict(
        dynamic_pars    = dynamic_pars,
        sequence        = sequence,
        change_beta     = change_beta,
        clip_edges      = clip_edges,
        test_num        = test_num,
        test_prob       = test_prob,
        contact_tracing = contact_tracing,
        )
    try:
        IntervClass = mapping[which]
    except:
        available = ', '.join(mapping.keys())
        errormsg = f'Only interventions "{available}" are available in dictionary representation, not "{which}"'
        raise sc.KeyNotFoundError(errormsg)
    intervention = IntervClass(**pars)
    return intervention


class Intervention:
    '''
    Base class for interventions. By default, interventions are printed using a
    dict format, which they can be recreated from. To display all the attributes
    of the intervention, use disp() instead.

    To retrieve a particular intervention from a sim, use sim.get_intervention().

    Args:
        label (str): a label for the intervention (used for plotting, and for ease of identification)
        show_label (bool): whether or not to include the label, if provided, in the legend
        do_plot (bool): whether or not to plot the intervention
        line_args (dict): arguments passed to pl.axvline() whe plotting
    '''
    def __init__(self, label=None, show_label=True, do_plot=None, line_args=None):
        self.label = label # e.g. "Close schools"
        self.show_label = show_label # Show the label by default
        self.do_plot = do_plot if do_plot is not None else True # Plot the intervention, including if None
        self.line_args = sc.mergedicts(dict(linestyle='--', c=[0,0,0]), line_args) # Do not set alpha by default due to the issue of overlapping interventions
        self.days = [] # The start and end days of the intervention
        self.initialized = False # Whether or not it has been initialized
        return


    def __repr__(self):
        ''' Return a JSON-friendly output if possible, else revert to pretty repr '''
        try:
            json = self.to_json()
            which = json['which']
            pars = json['pars']
            parstr = ', '.join([f'{k}={v}' for k,v in pars.items()])
            output = f"cv.{which}({parstr})"
        except Exception as E:
            output = type(self) + f' ({str(E)})' # If that fails, print why
        return output


    def disp(self):
        ''' Print a detailed representation of the intervention '''
        return print(sc.prepr(self))


    def _store_args(self):
        ''' Store the user-supplied arguments for later use in to_json '''
        f0 = inspect.currentframe() # This "frame", i.e. Intervention.__init__()
        f1 = inspect.getouterframes(f0) # The list of outer frames
        parent = f1[1].frame # The parent frame, e.g. change_beta.__init__()
        _,_,_,values = inspect.getargvalues(parent) # Get the values of the arguments
        self.input_args = {}
        for key,value in values.items():
            if key == 'kwargs': # Store additional kwargs directly
                for k2,v2 in value.items():
                    self.input_args[k2] = v2 # These are already a dict
            elif key not in ['self', '__class__']: # Everything else, but skip these
                self.input_args[key] = value
        return


    def initialize(self, sim):
        '''
        Initialize intervention -- this is used to make modifications to the intervention
        that can't be done until after the sim is created.
        '''
        self.initialized = True
        return


    def apply(self, sim):
        '''
        Apply the intervention. This is the core method which each drived intervention
        class must implement. This method gets called at each timestep and can make
        arbitrary changes to the Sim object, as well as storing or modifying the
        state of the intervention.

        Args:
            sim: the Sim instance

        Returns:
            None
        '''
        raise NotImplementedError


    def plot_intervention(self, sim, ax=None, **kwargs):
        '''
        Plot the intervention

        This can be used to do things like add vertical lines on days when
        interventions take place. Can be disabled by setting self.do_plot=False.

        Args:
            sim: the Sim instance
            ax: the axis instance
            kwargs: passed to ax.axvline()

        Returns:
            None
        '''
        line_args = sc.mergedicts(self.line_args, kwargs)
        if self.do_plot or self.do_plot is None:
            if ax is None:
                ax = pl.gca()
            for day in self.days:
                if day is not None:
                    if self.show_label: # Choose whether to include the label in the legend
                        label = self.label
                    else:
                        label = None
                    ax.axvline(day, label=label, **line_args)
        return


    def to_json(self):
        '''
        Return JSON-compatible representation

        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. In the first instance, the object dict will be returned.
        However, if an intervention itself contains non-standard variables as
        attributes, then its `to_json` method will need to handle those.

        Note that simply printing an intervention will usually return a representation
        that can be used to recreate it.

        Returns:
            JSON-serializable representation (typically a dict, but could be anything else)
        '''
        which = self.__class__.__name__
        pars = sc.jsonify(self.input_args)
        output = dict(which=which, pars=pars)
        return output


class dynamic_pars(Intervention):
    '''
    A generic intervention that modifies a set of parameters at specified points
    in time.

    The intervention takes a single argument, pars, which is a dictionary of which
    parameters to change, with following structure: keys are the parameters to change,
    then subkeys 'days' and 'vals' are either a scalar or list of when the change(s)
    should take effect and what the new value should be, respectively.

    You can also pass parameters to change directly as keyword arguments.

    Args:
        pars (dict): described above
        kwargs (dict): passed to Intervention()

    **Examples**::

        interv = cv.dynamic_pars(n_imports=dict(days=10, vals=100))
        interv = cv.dynamic_pars({'beta':{'days':[14, 28], 'vals':[0.005, 0.015]}, 'rel_death_prob':{'days':30, 'vals':2.0}}) # Change beta, and make diagnosed people stop transmitting

    '''

    def __init__(self, pars=None, **kwargs):

        # Find valid sim parameters and move matching keyword arguments to the pars dict
        pars = sc.mergedicts(pars) # Ensure it's a dictionary
        sim_par_keys = list(cvpar.make_pars().keys()) # Get valid sim parameters
        kwarg_keys = [k for k in kwargs.keys() if k in sim_par_keys]
        for kkey in kwarg_keys:
            pars[kkey] = kwargs.pop(kkey)

        # Do standard initialization
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated

        # Handle the rest of the initialization
        subkeys = ['days', 'vals']
        for parkey in pars.keys():
            for subkey in subkeys:
                if subkey not in pars[parkey].keys():
                    errormsg = f'Parameter {parkey} is missing subkey {subkey}'
                    raise sc.KeyNotFoundError(errormsg)
                if sc.isnumber(pars[parkey][subkey]): # Allow scalar values or dicts, but leave everything else unchanged
                    pars[parkey][subkey] = sc.promotetoarray(pars[parkey][subkey])
            len_days = len(pars[parkey]['days'])
            len_vals = len(pars[parkey]['vals'])
            if len_days != len_vals:
                raise ValueError(f'Length of days ({len_days}) does not match length of values ({len_vals}) for parameter {parkey}')
        self.pars = pars
        return


    def apply(self, sim):
        ''' Loop over the parameters, and then loop over the days, applying them if any are found '''
        t = sim.t
        for parkey,parval in self.pars.items():
            for ind in find_day(parval['days'], t):
                self.days.append(t)
                val = parval['vals'][ind]
                if isinstance(val, dict):
                    sim[parkey].update(val) # Set the parameter if a nested dict
                else:
                    sim[parkey] = val # Set the parameter if not a dict
        return


class sequence(Intervention):
    '''
    This is an example of a meta-intervention which switches between a sequence of interventions.

    Args:
        days (list): the days on which to start applying each intervention
        interventions (list): the interventions to apply on those days
        kwargs (dict): passed to Intervention()

    **Example**::

        interv = cv.sequence(days=[10, 51], interventions=[
                    cv.test_num(n_tests=[100]*npts),
                    cv.test_prob(symptomatic_prob=0.2, asymptomatic_prob=0.002),
                ])
    '''

    def __init__(self, days, interventions, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        assert len(days) == len(interventions)
        self.days = days
        self.interventions = interventions
        return


    def initialize(self, sim):
        ''' Fix the dates '''
        self.days = [sim.day(day) for day in self.days]
        self.days_arr = np.array(self.days + [sim.npts])
        for intervention in self.interventions:
            intervention.initialize(sim)
        self.initialized = True
        return


    def apply(self, sim):
        inds = find_day(self.days_arr <= sim.t, which='last')
        if len(inds):
            return self.interventions[inds[0]].apply(sim)



#%% Beta interventions

__all__+= ['change_beta', 'clip_edges']


def process_days(sim, days, return_dates=False):
    '''
    Ensure lists of days are in consistent format. Used by change_beta, clip_edges,
    and some analyzers. If day is 'end' or -1, use the final day of the simulation.
    Optionally return dates as well as days.
    '''
    if sc.isstring(days) or not sc.isiterable(days):
        days = sc.promotetolist(days)
    for d,day in enumerate(days):
        if day in ['end', -1]:
            day = sim['end_day']
        days[d] = sim.day(day) # Ensure it's an integer and not a string or something
    days = np.sort(sc.promotetoarray(days)) # Ensure they're an array and in order
    if return_dates:
        dates = [sim.date(day) for day in days] # Store as date strings
        return days, dates
    else:
        return days


def process_changes(sim, changes, days):
    '''
    Ensure lists of changes are in consistent format. Used by change_beta and clip_edges.
    '''
    changes = sc.promotetoarray(changes)
    if len(days) != len(changes):
        errormsg = f'Number of days supplied ({len(days)}) does not match number of changes ({len(changes)})'
        raise ValueError(errormsg)
    return changes


class change_beta(Intervention):
    '''
    The most basic intervention -- change beta (transmission) by a certain amount
    on a given day or days. This can be used to represent physical distancing (although
    clip_edges() is more appropriate for overall changes in mobility, e.g. school
    or workplace closures), as well as hand-washing, masks, and other behavioral
    changes that affect transmission rates.

    Args:
        days    (int/arr):   the day or array of days to apply the interventions
        changes (float/arr): the changes in beta (1 = no change, 0 = no transmission)
        layers  (str/list):  the layers in which to change beta (default: all)
        kwargs  (dict):      passed to Intervention()

    **Examples**::

        interv = cv.change_beta(25, 0.3) # On day 25, reduce overall beta by 70% to 0.3
        interv = cv.change_beta([14, 28], [0.7, 1], layers='s') # On day 14, reduce beta by 30%, and on day 28, return to 1 for schools
    '''

    def __init__(self, days, changes, layers=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        self.days       = sc.dcp(days)
        self.changes    = sc.dcp(changes)
        self.layers     = sc.dcp(layers)
        self.orig_betas = None
        return


    def initialize(self, sim):
        ''' Fix days and store beta '''
        self.days    = process_days(sim, self.days)
        self.changes = process_changes(sim, self.changes, self.days)
        self.layers  = sc.promotetolist(self.layers, keepnone=True)
        self.orig_betas = {}
        for lkey in self.layers:
            if lkey is None:
                self.orig_betas['overall'] = sim['beta']
            else:
                self.orig_betas[lkey] = sim['beta_layer'][lkey]

        self.initialized = True
        return


    def apply(self, sim):

        # If this day is found in the list, apply the intervention
        for ind in find_day(self.days, sim.t):
            for lkey,new_beta in self.orig_betas.items():
                new_beta = new_beta * self.changes[ind]
                if lkey == 'overall':
                    sim['beta'] = new_beta
                else:
                    sim['beta_layer'][lkey] = new_beta

        return


class clip_edges(Intervention):
    '''
    Isolate contacts by removing them from the simulation. Contacts are treated as
    "edges", and this intervention works by removing them from sim.people.contacts
    and storing them internally. When the intervention is over, they are moved back.
    This intervention has quite similar effects as change_beta(), but is more appropriate
    for modeling the effects of mobility reductions such as school and workplace
    closures. The main difference is that since clip_edges() actually removes contacts,
    it affects the number of people who would be traced and placed in quarantine
    if an individual tests positive. It also alters the structure of the network
    -- i.e., compared to a baseline case of 20 contacts and a 2% chance of infecting
    each, there are slightly different statistics for a beta reduction (i.e., 20 contacts
    and a 1% chance of infecting each) versus an edge clipping (i.e., 10 contacts
    and a 2% chance of infecting each).

    Args:
        days (int or array): the day or array of days to isolate contacts
        changes (float or array): the changes in the number of contacts (1 = no change, 0 = no contacts)
        layers (str or list): the layers in which to isolate contacts (if None, then all layers)
        kwargs (dict): passed to Intervention()

    **Examples**::

        interv = cv.clip_edges(25, 0.3) # On day 25, reduce overall contacts by 70% to 0.3
        interv = cv.clip_edges([14, 28], [0.7, 1], layers='w') # On day 14, remove 30% of school contacts, and on day 28, restore them
    '''

    def __init__(self, days, changes, layers=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        self.days     = sc.dcp(days)
        self.changes  = sc.dcp(changes)
        self.layers   = sc.dcp(layers)
        self.contacts = None
        return


    def initialize(self, sim):
        self.days    = process_days(sim, self.days)
        self.changes = process_changes(sim, self.changes, self.days)
        if self.layers is None:
            self.layers = sim.layer_keys()
        else:
            self.layers = sc.promotetolist(self.layers)
        self.contacts = cvb.Contacts(layer_keys=self.layers)
        self.initialized = True
        return


    def apply(self, sim):

        # If this day is found in the list, apply the intervention
        for ind in find_day(self.days, sim.t):

            # Do the contact moving
            for lkey in self.layers:
                s_layer = sim.people.contacts[lkey] # Contact layer in the sim
                i_layer = self.contacts[lkey] # Contat layer in the intervention
                n_sim = len(s_layer) # Number of contacts in the simulation layer
                n_int = len(i_layer) # Number of contacts in the intervention layer
                n_contacts = n_sim + n_int # Total number of contacts
                if n_contacts:
                    current_prop = n_sim/n_contacts # Current proportion of contacts in the sim, e.g. 1.0 initially
                    desired_prop = self.changes[ind] # Desired proportion, e.g. 0.5
                    prop_to_move = current_prop - desired_prop # Calculate the proportion of contacts to move
                    n_to_move = int(prop_to_move*n_contacts) # Number of contacts to move
                    from_sim = (n_to_move>0) # Check if we're moving contacts from the sim
                    if from_sim: # We're moving from the sim to the intervention
                        inds = cvu.choose(max_n=n_sim, n=n_to_move)
                        to_move = s_layer.pop_inds(inds)
                        i_layer.append(to_move)
                    else: # We're moving from the intervention back to the sim
                        inds = cvu.choose(max_n=n_int, n=abs(n_to_move))
                        to_move = i_layer.pop_inds(inds)
                        s_layer.append(to_move)
                else:
                    print(f'Warning: clip_edges() was applied to layer "{lkey}", but no edges were found; please check sim.people.contacts["{lkey}"]')

        # Ensure the edges get deleted at the end
        if sim.t == sim.tvec[-1]:
            self.contacts = None # Reset to save memory

        return



#%% Testing interventions

__all__+= ['test_num', 'test_prob', 'contact_tracing']


def process_daily_data(daily_data, sim, start_day, as_int=False):
    '''
    This function performs one of three things: if the daily test data are supplied as
    a number, then it converts it to an array of the right length. If the daily
    data are supplied as a Pandas series or dataframe with a date index, then it
    reindexes it to match the start date of the simulation. If the daily data are
    supplied as a string, then it will convert it to a column and try to read from
    that. Otherwise, it does nothing.

    Args:
        daily_data (str, number, dataframe, or series): the data to convert to standardized format
        sim (Sim): the simulation object
        start_day (date): the start day of the simulation, in already-converted datetime.date format
        as_int (bool): whether to convert to an integer
    '''
    # Handle string arguments
    if sc.isstring(daily_data):
        if daily_data == 'data':
            daily_data = sim.data['new_tests'] # Use default name
        else:
            try:
                daily_data = sim.data[daily_data]
            except Exception as E:
                errormsg = f'Tried to load testing data from sim.data["{daily_data}"], but that failed: {str(E)}.\nPlease ensure data are loaded into the sim and the column exists.'
                raise ValueError(errormsg) from E

    # Handle other arguments
    if sc.isnumber(daily_data):  # If a number, convert to an array
        if as_int: daily_data = int(daily_data) # Make it an integer
        daily_data = np.array([daily_data] * sim.npts)
    elif isinstance(daily_data, (pd.Series, pd.DataFrame)):
        start_date = sim['start_day'] + dt.timedelta(days=start_day)
        end_date = daily_data.index[-1]
        dateindex = pd.date_range(start_date, end_date)
        daily_data = daily_data.reindex(dateindex, fill_value=0).to_numpy()

    return daily_data


def get_subtargets(subtarget, sim):
    '''
    A small helper function to see if subtargeting is a list of indices to use,
    or a function that needs to be called. If a function, it must take a single
    argument, a sim object, and return a list of indices. Also validates the values.
    Currently designed for use with testing interventions, but could be generalized
    to other interventions. Not typically called directly by the user.

    Args:
        subtarget (dict): dict with keys 'inds' and 'vals'; see test_num() for examples of a valid subtarget dictionary
        sim (Sim): the simulation object
    '''

    # Validation
    if callable(subtarget):
        subtarget = subtarget(sim)

    if 'inds' not in subtarget:
        errormsg = f'The subtarget dict must have keys "inds" and "vals", but you supplied {subtarget}'
        raise ValueError(errormsg)

    # Handle the two options of type
    if callable(subtarget['inds']): # A function has been provided
        subtarget_inds = subtarget['inds'](sim) # Call the function to get the indices
    else:
        subtarget_inds = subtarget['inds'] # The indices are supplied directly

    # Validate the values
    if callable(subtarget['vals']): # A function has been provided
        subtarget_vals = subtarget['vals'](sim) # Call the function to get the indices
    else:
        subtarget_vals = subtarget['vals'] # The indices are supplied directly
    if sc.isiterable(subtarget_vals):
        if len(subtarget_vals) != len(subtarget_inds):
            errormsg = f'Length of subtargeting indices ({len(subtarget_inds)}) does not match length of values ({len(subtarget_vals)})'
            raise ValueError(errormsg)

    return subtarget_inds, subtarget_vals


def get_quar_inds(quar_policy, sim):
    '''
    Helper function to return the appropriate indices for people in quarantine
    based on the current quarantine testing "policy". Used by test_num and test_prob.
    Not for use by the user.

    If quar_policy is a number or a list of numbers, then it is interpreted as
    the number of days after the start of quarantine when a test is performed.
    It can also be a function that returns the list of indices.

    Args:
        quar_policy (str, int, list, func): 'start', people entering quarantine; 'end', people leaving; 'both', entering and leaving; 'daily', every day in quarantine
        sim (Sim): the simulation object
    '''
    t = sim.t
    if   quar_policy is None:    quar_test_inds = np.array([])
    elif quar_policy == 'start': quar_test_inds = cvu.true(sim.people.date_quarantined==t-1) # Actually do the day after since testing usually happens before contact tracing
    elif quar_policy == 'end':   quar_test_inds = cvu.true(sim.people.date_end_quarantine==t+1) # +1 since they are released on date_end_quarantine, so do the day before
    elif quar_policy == 'both':  quar_test_inds = np.concatenate([cvu.true(sim.people.date_quarantined==t-1), cvu.true(sim.people.date_end_quarantine==t+1)])
    elif quar_policy == 'daily': quar_test_inds = cvu.true(sim.people.quarantined)
    elif sc.isnumber(quar_policy) or (sc.isiterable(quar_policy) and not sc.isstring(quar_policy)):
        quar_policy = sc.promotetoarray(quar_policy)
        quar_test_inds = np.unique(np.concatenate([cvu.true(sim.people.date_quarantined==t-1-q) for q in quar_policy]))
    elif callable(quar_policy):
        quar_test_inds = quar_policy(sim)
    else:
        errormsg = f'Quarantine policy "{quar_policy}" not recognized: must be a string (start, end, both, daily), int, list, array, set, tuple, or function'
        raise ValueError(errormsg)
    return quar_test_inds


class test_num(Intervention):
    '''
    Test the specified number of people per day. Useful for including historical
    testing data. The probability of a given person getting a test is dependent
    on the total number of tests, population size, and odds ratios. Compare this
    intervention with cv.test_prob().

    Args:
        daily_tests (arr)   : number of tests per day, can be int, array, or dataframe/series; if integer, use that number every day; if 'data' or another string, use loaded data
        symp_test   (float) : odds ratio of a symptomatic person testing (default: 100x more likely)
        quar_test   (float) : probability of a person in quarantine testing (default: no more likely)
        quar_policy (str)   : policy for testing in quarantine: options are 'start' (default), 'end', 'both' (start and end), 'daily'; can also be a number or a function, see get_quar_inds()
        subtarget   (dict)  : subtarget intervention to people with particular indices (format: {'ind': array of indices, or function to return indices from the sim, 'vals': value(s) to apply}
        ili_prev    (arr)   : prevalence of influenza-like-illness symptoms in the population; can be float, array, or dataframe/series
        sensitivity (float) : test sensitivity (default 100%, i.e. no false negatives)
        loss_prob   (float) : probability of the person being lost-to-follow-up (default 0%, i.e. no one lost to follow-up)
        test_delay  (int)   : days for test result to be known (default 0, i.e. results available instantly)
        start_day   (int)   : day the intervention starts (default: 0, i.e. first day of the simulation)
        end_day     (int)   : day the intervention ends
        swab_delay  (dict)  : distribution for the delay from onset to swab; if this is present, it is used instead of test_delay
        kwargs      (dict)  : passed to Intervention()

    **Examples**::

        interv = cv.test_num(daily_tests=[0.10*n_people]*npts)
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': cv.true(sim.people.age>50), 'vals': 1.2}) # People over 50 are 20% more likely to test
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': lambda sim: cv.true(sim.people.age>50), 'vals': 1.2}) # People over 50 are 20% more likely to test
        interv = cv.test_num(daily_tests='data') # Take number of tests from loaded data using default column name (new_tests)
        interv = cv.test_num(daily_tests='swabs_per_day') # Take number of tests from loaded data using a custom column name
    '''

    def __init__(self, daily_tests, symp_test=100.0, quar_test=1.0, quar_policy=None, subtarget=None,
                 ili_prev=None, sensitivity=1.0, loss_prob=0, test_delay=0,
                 start_day=0, end_day=None, swab_delay=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        self.daily_tests = daily_tests # Should be a list of length matching time
        self.symp_test   = symp_test   # Set probability of testing symptomatics
        self.quar_test   = quar_test # Probability of testing people in quarantine
        self.quar_policy = quar_policy if quar_policy else 'start'
        self.subtarget   = subtarget  # Set any other testing criteria
        self.ili_prev    = ili_prev     # Should be a list of length matching time or a float or a dataframe
        self.sensitivity = sensitivity
        self.loss_prob   = loss_prob
        self.test_delay  = test_delay
        self.start_day   = start_day
        self.end_day     = end_day
        self.pdf         = cvu.get_pdf(**sc.mergedicts(swab_delay)) # If provided, get the distribution's pdf -- this returns an empty dict if None is supplied
        return


    def initialize(self, sim):
        ''' Fix the dates and number of tests '''

        # Handle days
        self.start_day   = sim.day(self.start_day)
        self.end_day     = sim.day(self.end_day)
        self.days        = [self.start_day, self.end_day]

        # Process daily data
        self.daily_tests = process_daily_data(self.daily_tests, sim, self.start_day)
        self.ili_prev    = process_daily_data(self.ili_prev,    sim, self.start_day)

        self.initialized = True

        return


    def apply(self, sim):

        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        # Check that there are still tests
        rel_t = t - self.start_day
        if rel_t < len(self.daily_tests):
            n_tests = sc.randround(self.daily_tests[rel_t]/sim.rescale_vec[t]) # Correct for scaling that may be applied by rounding to the nearest number of tests
            if not (n_tests and pl.isfinite(n_tests)): # If there are no tests today, abort early
                return
            else:
                sim.results['new_tests'][t] += n_tests
        else:
            return

        test_probs = np.ones(sim.n) # Begin by assigning equal testing weight (converted to a probability) to everyone

        # Calculate test probabilities for people with symptoms
        symp_inds = cvu.true(sim.people.symptomatic)
        symp_test = self.symp_test
        if self.pdf: # Handle the onset to swab delay
            symp_time = cvd.default_int(t - sim.people.date_symptomatic[symp_inds]) # Find time since symptom onset
            inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
            count = np.nan * np.ones(inv_count.shape) # Initialize the count
            count[inv_count != 0] = 1/inv_count[inv_count != 0] # Update the counts where defined
            symp_test *= self.pdf.pdf(symp_time) * count[symp_time] # Put it all together

        test_probs[symp_inds] *= symp_test # Update the test probabilities

        # Handle symptomatic testing, taking into account prevalence of ILI symptoms
        if self.ili_prev is not None:
            if rel_t < len(self.ili_prev):
                n_ili = int(self.ili_prev[rel_t] * sim['pop_size'])  # Number with ILI symptoms on this day
                ili_inds = cvu.choose(sim['pop_size'], n_ili) # Give some people some symptoms. Assuming that this is independent of COVID symptomaticity...
                ili_inds = np.setdiff1d(ili_inds, symp_inds)
                test_probs[ili_inds] *= self.symp_test

        # Handle quarantine testing
        quar_test_inds = get_quar_inds(self.quar_policy, sim)
        test_probs[quar_test_inds] *= self.quar_test

        # Handle any other user-specified testing criteria
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            test_probs[subtarget_inds] = test_probs[subtarget_inds]*subtarget_vals

        # Don't re-diagnose people
        diag_inds  = cvu.true(sim.people.diagnosed)
        test_probs[diag_inds] = 0.0

        # With dynamic rescaling, we have to correct for uninfected people outside of the population who would test
        if sim.rescale_vec[t]/sim['pop_scale'] < 1: # We still have rescaling to do
            in_pop_tot_prob = test_probs.sum()*sim.rescale_vec[t] # Total "testing weight" of people in the subsampled population
            out_pop_tot_prob = sim.scaled_pop_size - sim.rescale_vec[t]*sim['pop_size'] # Find out how many people are missing and assign them each weight 1
            in_frac = in_pop_tot_prob/(in_pop_tot_prob + out_pop_tot_prob) # Fraction of tests which should fall in the sample population
            n_tests = sc.randround(n_tests*in_frac) # Recompute the number of tests

        # Now choose who gets tested and test them
        n_tests = min(n_tests, (test_probs!=0).sum()) # Don't try to test more people than have nonzero testing probability
        test_inds = cvu.choose_w(probs=test_probs, n=n_tests, unique=True) # Choose who actually tests
        sim.people.test(test_inds, test_sensitivity=self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)

        return test_inds


class test_prob(Intervention):
    '''
    Assign each person a probability of being tested for COVID based on their
    symptom state, quarantine state, and other states. Unlike test_num, the
    total number of tests not specified, but rather is an output.

    Args:
        symp_prob        (float)     : probability of testing a symptomatic (unquarantined) person
        asymp_prob       (float)     : probability of testing an asymptomatic (unquarantined) person (default: 0)
        symp_quar_prob   (float)     : probability of testing a symptomatic quarantined person (default: same as symp_prob)
        asymp_quar_prob  (float)     : probability of testing an asymptomatic quarantined person (default: same as asymp_prob)
        quar_policy      (str)       : policy for testing in quarantine: options are 'start' (default), 'end', 'both' (start and end), 'daily'; can also be a number or a function, see get_quar_inds()
        subtarget        (dict)      : subtarget intervention to people with particular indices  (see test_num() for details)
        ili_prev         (float/arr) : prevalence of influenza-like-illness symptoms in the population; can be float, array, or dataframe/series
        sensitivity      (float)     : test sensitivity (default 100%, i.e. no false negatives)
        loss_prob        (float)     : probability of the person being lost-to-follow-up (default 0%, i.e. no one lost to follow-up)
        test_delay       (int)       : days for test result to be known (default 0, i.e. results available instantly)
        start_day        (int)       : day the intervention starts (default: 0, i.e. first day of the simulation)
        end_day          (int)       : day the intervention ends (default: no end)
        kwargs           (dict)      : passed to Intervention()

    **Examples**::

        interv = cv.test_prob(symp_prob=0.1, asymp_prob=0.01) # Test 10% of symptomatics and 1% of asymptomatics
        interv = cv.test_prob(symp_quar_prob=0.4) # Test 40% of those in quarantine with symptoms
    '''
    def __init__(self, symp_prob, asymp_prob=0.0, symp_quar_prob=None, asymp_quar_prob=None, quar_policy=None, subtarget=None, ili_prev=None,
                 sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0, end_day=None, swab_delay=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        self.symp_prob        = symp_prob
        self.asymp_prob       = asymp_prob
        self.symp_quar_prob   = symp_quar_prob  if  symp_quar_prob is not None else  symp_prob
        self.asymp_quar_prob  = asymp_quar_prob if asymp_quar_prob is not None else asymp_prob
        self.quar_policy      = quar_policy if quar_policy else 'start'
        self.subtarget        = subtarget
        self.ili_prev         = ili_prev
        self.sensitivity      = sensitivity
        self.loss_prob        = loss_prob
        self.test_delay       = test_delay
        self.start_day        = start_day
        self.end_day          = end_day
        self.pdf              = cvu.get_pdf(**sc.mergedicts(swab_delay)) # If provided, get the distribution's pdf -- this returns an empty dict if None is supplied
        return


    def initialize(self, sim):
        ''' Fix the dates '''
        self.start_day = sim.day(self.start_day)
        self.end_day   = sim.day(self.end_day)
        self.days      = [self.start_day, self.end_day]
        self.ili_prev  = process_daily_data(self.ili_prev, sim, self.start_day)
        self.initialized = True
        return


    def apply(self, sim):
        ''' Perform testing '''
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        # Find probablity for symptomatics to be tested
        symp_inds  = cvu.true(sim.people.symptomatic)
        symp_prob = self.symp_prob
        if self.pdf:
            symp_time = cvd.default_int(t - sim.people.date_symptomatic[symp_inds]) # Find time since symptom onset
            inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
            count = np.nan * np.ones(inv_count.shape)
            count[inv_count != 0] = 1/inv_count[inv_count != 0]
            symp_prob = np.ones(len(symp_time))
            inds = 1 > (symp_time*self.symp_prob)
            symp_prob[inds] = self.symp_prob/(1-symp_time[inds]*self.symp_prob)
            symp_prob = self.pdf.pdf(symp_time) * symp_prob * count[symp_time]

        # Define symptomatics, accounting for ILI prevalence
        pop_size = sim['pop_size']
        ili_inds = []
        if self.ili_prev is not None:
            rel_t = t - self.start_day
            if rel_t < len(self.ili_prev):
                n_ili = int(self.ili_prev[rel_t] * pop_size)  # Number with ILI symptoms on this day
                ili_inds = cvu.choose(pop_size, n_ili) # Give some people some symptoms, assuming that this is independent of COVID symptomaticity...
                ili_inds = np.setdiff1d(ili_inds, symp_inds)

        # Define asymptomatics: those who neither have COVID symptoms nor ILI symptoms
        asymp_inds = np.setdiff1d(np.setdiff1d(np.arange(pop_size), symp_inds), ili_inds)

        # Handle quarantine and other testing criteria
        quar_test_inds = get_quar_inds(self.quar_policy, sim)
        symp_quar_inds  = np.intersect1d(quar_test_inds, symp_inds)
        asymp_quar_inds = np.intersect1d(quar_test_inds, asymp_inds)
        diag_inds       = cvu.true(sim.people.diagnosed)
        if self.subtarget is not None:
            subtarget_inds  = self.subtarget['inds']

        # Construct the testing probabilities piece by piece -- complicated, since need to do it in the right order
        test_probs = np.zeros(sim.n) # Begin by assigning equal testing probability to everyone
        test_probs[symp_inds]       = symp_prob            # People with symptoms (true positive)
        test_probs[ili_inds]        = symp_prob            # People with symptoms (false positive)
        test_probs[asymp_inds]      = self.asymp_prob      # People without symptoms
        test_probs[symp_quar_inds]  = self.symp_quar_prob  # People with symptoms in quarantine
        test_probs[asymp_quar_inds] = self.asymp_quar_prob # People without symptoms in quarantine
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            test_probs[subtarget_inds] = subtarget_vals # People being explicitly subtargeted
        test_probs[diag_inds] = 0.0 # People who are diagnosed don't test
        test_inds = cvu.true(cvu.binomial_arr(test_probs)) # Finally, calculate who actually tests

        # Actually test people
        sim.people.test(test_inds, test_sensitivity=self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay) # Actually test people
        sim.results['new_tests'][t] += int(len(test_inds)*sim['pop_scale']/sim.rescale_vec[t]) # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec

        return test_inds


class contact_tracing(Intervention):
    '''
    Contact tracing of people who are diagnosed. When a person is diagnosed positive
    (by either test_num() or test_prob(); this intervention has no effect if there
    is not also a testing intervention active), a certain proportion of the index
    case's contacts (defined by trace_prob) are contacted after a certain number
    of days (defined by trace_time). After they are contacted, they are placed
    into quarantine (with effectiveness quar_factor, a simulation parameter) for
    a certain period (defined by quar_period, another simulation parameter). They
    may also change their testing probability, if test_prob() is defined.

    Args:
        trace_probs (float/dict): probability of tracing, per layer (default: 100%, i.e. everyone is traced)
        trace_time  (float/dict): days required to trace, per layer (default: 0, i.e. no delay)
        start_day   (int):        intervention start day (default: 0, i.e. the start of the simulation)
        end_day     (int):        intervention end day (default: no end)
        presumptive (bool):       whether or not to begin isolation and contact tracing on the presumption of a positive diagnosis (default: no)
        quar_period (int):        number of days to quarantine when notified as a known contact. Default value is pars['quar_period']
        kwargs      (dict):       passed to Intervention()

    **Example**::

        tp = cv.test_prob(symp_prob=0.1, asymp_prob=0.01)
        ct = cv.contact_tracing(trace_probs=0.5, trace_time=2)
        sim = cv.Sim(interventions=[tp, ct]) # Note that without testing, contact tracing has no effect
    '''
    def __init__(self, trace_probs=None, trace_time=None, start_day=0, end_day=None, presumptive=False, quar_period=None,  **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        self.trace_probs = trace_probs
        self.trace_time  = trace_time
        self.start_day   = start_day
        self.end_day     = end_day
        self.presumptive = presumptive
        self.quar_period = quar_period  #: If quar_period is None, it will be drawn from sim.pars at initialization
        return


    def initialize(self, sim):
        ''' Process the dates and dictionaries '''
        self.start_day = sim.day(self.start_day)
        self.end_day   = sim.day(self.end_day)
        self.days      = [self.start_day, self.end_day]
        if self.trace_probs is None:
            self.trace_probs = 1.0
        if self.trace_time is None:
            self.trace_time = 0.0
        if self.quar_period is None:
            self.quar_period = sim.pars['quar_period']
        if sc.isnumber(self.trace_probs):
            val = self.trace_probs
            self.trace_probs = {k:val for k in sim.people.layer_keys()}
        if sc.isnumber(self.trace_time):
            val = self.trace_time
            self.trace_time = {k:val for k in sim.people.layer_keys()}
        self.initialized = True
        return


    def apply(self, sim):
        '''
        Trace and notify contacts

        Tracing involves three steps that can independently be overloaded or extended
        by derived classes

        - Select which confirmed cases get interviewed by contact tracers
        - Identify the contacts of the confirmed case
        - Notify those contacts that they have been exposed and need to take some action
        '''
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        trace_inds = self.select_cases(sim)
        contacts = self.identify_contacts(sim, trace_inds)
        self.notify_contacts(sim, contacts)
        return contacts


    def select_cases(self, sim):
        '''
        Return people to be traced at this time step
        '''
        if not self.presumptive:
            inds = cvu.true(sim.people.date_diagnosed == sim.t) # Diagnosed this time step, time to trace
        else:
            just_tested = cvu.true(sim.people.date_tested == sim.t) # Tested this time step, time to trace
            inds = cvu.itruei(sim.people.exposed, just_tested) # This is necessary to avoid infinite chains of asymptomatic testing
        return inds


    def identify_contacts(self, sim, trace_inds):
        '''
        Return contacts to notify by trace time

        In the base class, the trace time is the same per-layer, but derived classes might
        provide different functionality e.g. sampling the trace time from a distribution. The
        return value of this method is a dict keyed by trace time so that the `Person` object
        can be easily updated in `contact_tracing.notify_contacts`

        Args:
            sim: Simulation object
            trace_inds: Indices of people to trace

        Returns: {trace_time: np.array(inds)} dictionary storing which people to notify
        '''

        if not len(trace_inds):
            return dict()

        contacts = defaultdict(list)

        for lkey, this_trace_prob in self.trace_probs.items():

            if this_trace_prob == 0:
                continue

            traceable_inds = sim.people.contacts[lkey].find_contacts(trace_inds)
            if len(traceable_inds):
                contacts[self.trace_time[lkey]].extend(cvu.binomial_filter(this_trace_prob, traceable_inds)) # Filter the indices according to the probability of being able to trace this layer

        array_contacts = {}
        for trace_time, inds in contacts.items():
            array_contacts[trace_time] = np.fromiter(inds, dtype=cvd.default_int)

        return array_contacts


    def notify_contacts(self, sim, contacts):
        '''
        Notify contacts

        This method represents notifying people that they have had contact with a confirmed case.
        In this base class, that involves

        - Setting the 'known_contact' flag and recording the 'date_known_contact'
        - Scheduling quarantine

        Args:
            sim: Simulation object
            contacts: {trace_time: np.array(inds)} dictionary storing which people to notify
        '''
        for trace_time, contact_inds in contacts.items():
            sim.people.known_contact[contact_inds] = True
            sim.people.date_known_contact[contact_inds] = np.fmin(sim.people.date_known_contact[contact_inds], sim.t + trace_time)
            sim.people.schedule_quarantine(contact_inds, start_date=sim.t + trace_time, period=self.quar_period - trace_time)  # Schedule quarantine for the notified people to start on the date they will be notified
        return



#%% Treatment and prevention interventions

__all__+= ['vaccine']


class vaccine(Intervention):
    '''
    Apply a vaccine to a subset of the population. In addition to changing the
    relative susceptibility and the probability of developing symptoms if still
    infected, this sintervention stores several types of data:

        - ``vaccinations``:      the number of vaccine doses per person
        - ``vaccination_dates``: list of dates per person
        - ``orig_rel_sus``:      relative susceptibility per person at the beginning of the simulation
        - ``orig_symp_prob``:    probability of developing symptoms per person at the beginning of the simulation
        - ``mod_rel_sus``:       modifier on default susceptibility due to the vaccine
        - ``mod_symp_prob``:     modifier on default symptom probability due to the vaccine

    Args:
        days (int or array): the day or array of days to apply the interventions
        prob      (float): probability of being vaccinated (i.e., fraction of the population)
        rel_sus   (float): relative change in susceptibility; 0 = perfect, 1 = no effect
        rel_symp  (float): relative change in symptom probability for people who still get infected; 0 = perfect, 1 = no effect
        subtarget  (dict): subtarget intervention to people with particular indices (see test_num() for details)
        cumulative (bool): whether cumulative doses have cumulative effects (default false); can also be an array for efficacy per dose, with the last entry used for multiple doses; thus True = [1] and False = [1,0]
        kwargs     (dict): passed to Intervention()

    **Examples**::

        interv = cv.vaccine(days=50, prob=0.3, rel_sus=0.5, rel_symp=0.1)
        interv = cv.vaccine(days=[10,20,30,40], prob=0.8, rel_sus=0.5, cumulative=[1, 0.3, 0.1, 0]) # A vaccine with efficacy up to the 3rd dose
    '''
    def __init__(self, days, prob=1.0, rel_sus=0.0, rel_symp=0.0, subtarget=None, cumulative=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        self.days      = sc.dcp(days)
        self.prob      = prob
        self.rel_sus   = rel_sus
        self.rel_symp  = rel_symp
        self.subtarget = subtarget
        if cumulative in [0, False]:
            cumulative = [1,0] # First dose has full efficacy, second has none
        elif cumulative in [1, True]:
            cumulative = [1] # All doses have full efficacy
        self.cumulative = np.array(cumulative, dtype=cvd.default_float) # Ensure it's an array
        return


    def initialize(self, sim):
        ''' Fix the dates and store the vaccinations '''
        self.days = process_days(sim, self.days)
        self.vaccinations      = np.zeros(sim.n, dtype=cvd.default_int) # Number of doses given per person
        self.vaccination_dates = [[] for p in range(sim.n)] # Store the dates when people are vaccinated
        self.orig_rel_sus      = sc.dcp(sim.people.rel_sus) # Keep a copy of pre-vaccination susceptibility
        self.orig_symp_prob    = sc.dcp(sim.people.symp_prob) # ...and symptom probability
        self.mod_rel_sus       = np.ones(sim.n, dtype=cvd.default_float) # Store the final modifiers
        self.mod_symp_prob     = np.ones(sim.n, dtype=cvd.default_float) # Store the final modifiers
        self.initialized = True
        return


    def apply(self, sim):
        ''' Perform vaccination '''

        # If this day is found in the list, apply the intervention
        for ind in find_day(self.days, sim.t):

            # Construct the testing probabilities piece by piece -- complicated, since need to do it in the right order
            vacc_probs = np.full(sim.n, self.prob) # Begin by assigning equal testing probability to everyone
            if self.subtarget is not None:
                subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
                vacc_probs[subtarget_inds] = subtarget_vals # People being explicitly subtargeted
            vacc_inds = cvu.true(cvu.binomial_arr(vacc_probs)) # Calculate who actually gets vaccinated

            # Calculate the effect per person
            vacc_doses = self.vaccinations[vacc_inds] # Calculate current doses
            eff_doses = np.minimum(vacc_doses, len(self.cumulative)-1) # Convert to a valid index
            vacc_eff = self.cumulative[eff_doses] # Pull out corresponding effect sizes
            rel_sus_eff  = (1.0 - vacc_eff) + vacc_eff*self.rel_sus
            rel_symp_eff = (1.0 - vacc_eff) + vacc_eff*self.rel_symp

            # Apply the vaccine to people
            sim.people.rel_sus[vacc_inds]   *= rel_sus_eff
            sim.people.symp_prob[vacc_inds] *= rel_symp_eff

            # Update counters
            self.mod_rel_sus[vacc_inds]   *= rel_sus_eff
            self.mod_symp_prob[vacc_inds] *= rel_symp_eff
            self.vaccinations[vacc_inds] += 1
            for v_ind in vacc_inds:
                self.vaccination_dates[v_ind].append(sim.t)

        return
