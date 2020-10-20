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
from . import misc as cvm


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
    all_inds = sc.findinds(val1=arr, val2=t)
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
            output = f"cv.InterventionDict('{which}', pars={pars})"
        except:
            output = sc.prepr(self)
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


    def plot(self, sim, ax=None, **kwargs):
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

    Args:
        pars (dict): described above
        kwargs (dict): passed to Intervention()

    **Examples**::

        interv = cv.dynamic_pars({'beta':{'days':[14, 28], 'vals':[0.005, 0.015]}, 'rel_death_prob':{'days':30, 'vals':2.0}}) # Change beta, and make diagnosed people stop transmitting
    '''

    def __init__(self, pars, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
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
        for ind in find_day(self.days_arr <= sim.t, which='last'):
            self.interventions[ind].apply(sim)
        return



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
    This function performs one of two things: if the daily data are supplied as
    a number, then it converts it to an array of the right length. If the daily
    data are supplied as a Pandas series or dataframe with a date index, then it
    reindexes it to match the start date of the simulation. Otherwise, it does
    nothing.

    Args:
        daily_data (number, dataframe, or series): the data to convert to standardized format
        sim (Sim): the simulation object
        start_day (date): the start day of the simulation, in already-converted datetime.date format
        as_int (bool): whether to convert to an integer
    '''
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
        daily_tests (arr)   : number of tests per day, can be int, array, or dataframe/series; if integer, use that number every day
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
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': sim.people.age>50, 'vals': 1.2}) # People over 50 are 20% more likely to test
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': lambda sim: sim.people.age>50, 'vals': 1.2}) # People over 50 are 20% more likely to test
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
            n_tests = cvu.randround(self.daily_tests[rel_t]/sim.rescale_vec[t]) # Correct for scaling that may be applied by rounding to the nearest number of tests
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
        diag_inds = cvu.true(sim.people.diagnosed)
        test_probs[diag_inds] = 0.0

        # With dynamic rescaling, we have to correct for uninfected people outside of the population who would test
        if sim.rescale_vec[t]/sim['pop_scale'] < 1: # We still have rescaling to do
            in_pop_tot_prob = test_probs.sum()*sim.rescale_vec[t] # Total "testing weight" of people in the subsampled population
            out_pop_tot_prob = sim.scaled_pop_size - sim.rescale_vec[t]*sim['pop_size'] # Find out how many people are missing and assign them each weight 1
            in_frac = in_pop_tot_prob/(in_pop_tot_prob + out_pop_tot_prob) # Fraction of tests which should fall in the sample population
            n_tests = cvu.randround(n_tests*in_frac) # Recompute the number of tests

        # Now choose who gets tested and test them
        n_tests = min(n_tests, (test_probs!=0).sum()) # Don't try to test more people than have nonzero testing probability
        test_inds = cvu.choose_w(probs=test_probs, n=n_tests, unique=True) # Choose who actually tests
        sim.people.test(test_inds, self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)

        return


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
        test_sensitivity (float)     : test sensitivity (default 100%, i.e. no false negatives)
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
                 test_sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0, end_day=None, swab_delay=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        self.symp_prob        = symp_prob
        self.asymp_prob       = asymp_prob
        self.symp_quar_prob   = symp_quar_prob  if  symp_quar_prob is not None else  symp_prob
        self.asymp_quar_prob  = asymp_quar_prob if asymp_quar_prob is not None else asymp_prob
        self.quar_policy      = quar_policy if quar_policy else 'start'
        self.subtarget        = subtarget
        self.ili_prev         = ili_prev
        self.test_sensitivity = test_sensitivity
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
        sim.people.test(test_inds, test_sensitivity=self.test_sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay) # Actually test people
        sim.results['new_tests'][t] += int(len(test_inds)*sim['pop_scale']/sim.rescale_vec[t]) # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec

        return


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
        kwargs      (dict):       passed to Intervention()

    **Example**::

        tp = cv.test_prob(symp_prob=0.1, asymp_prob=0.01)
        ct = cv.contact_tracing(trace_probs=0.5, trace_time=2)
        sim = cv.Sim(interventions=[tp, ct]) # Note that without testing, contact tracing has no effect
    '''
    def __init__(self, trace_probs=None, trace_time=None, start_day=0, end_day=None, presumptive=False, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        self.trace_probs = trace_probs
        self.trace_time  = trace_time
        self.start_day   = start_day
        self.end_day     = end_day
        self.presumptive = presumptive
        return


    def initialize(self, sim):
        ''' Fix the dates and dictionaries '''
        self.start_day = sim.day(self.start_day)
        self.end_day   = sim.day(self.end_day)
        self.days      = [self.start_day, self.end_day]
        if self.trace_probs is None:
            self.trace_probs = 1.0
        if self.trace_time is None:
            self.trace_time = 0.0
        if sc.isnumber(self.trace_probs):
            val = self.trace_probs
            self.trace_probs = {k:val for k in sim.people.layer_keys()}
        if sc.isnumber(self.trace_time):
            val = self.trace_time
            self.trace_time = {k:val for k in sim.people.layer_keys()}
        self.initialized = True
        return


    def apply(self, sim):
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        # Figure out whom to test and trace
        if not self.presumptive:
            trace_from_inds = cvu.true(sim.people.date_diagnosed == t) # Diagnosed this time step, time to trace
        else:
            just_tested = cvu.true(sim.people.date_tested == t) # Tested this time step, time to trace
            trace_from_inds = cvu.itruei(sim.people.exposed, just_tested) # This is necessary to avoid infinite chains of asymptomatic testing

        if len(trace_from_inds): # If there are any just-diagnosed people, go trace their contacts
            sim.people.trace(trace_from_inds, self.trace_probs, self.trace_time)

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


#%% School interventions

__all__ += ['set_rel_trans', 'close_schools', 'reopen_schools']


class set_rel_trans(Intervention):
    '''
    Sets the relative transmission factor on a given day, for a given age group

    Args:
        start_day : Day to set the transmission
        age: Age to adjust
        changes: Factor applied to rel_trans
    '''

    def __init__(self, start_day=None, age=None, changes=None, **kwargs):
        super().__init__(**kwargs)
        self._store_args()
        self.start_day   = start_day
        self.age         = age
        self.changes     = changes
        return

    def initialize(self, sim):
        self.start_day = sim.day(self.start_day)
        self.initialized = True

    def apply(self, sim):
        t = sim.t

        if t == self.start_day:
            under_age = [i for i in range(len(sim.people.age)) if sim.people.age[i] <= self.age]
            for ind in under_age:
                sim.people.rel_trans[ind] = sim.people.rel_trans[ind] * self.changes


class close_schools(Intervention):
    '''
        Shuts down and then reopens schools (possibly with a different network), by type and start day.

        Args:
            start_day   (dict)              : dictionary with school type as key and value is start_day, or string containing start day for the 's' layer
            day_schools_closed (str)        : day to close school layer
            pop_file (People object)        : People object with different network structure
            kwargs      (dict)              : passed to Intervention

        **Examples**
            iterv = cv.close_schools(day_schools_closed = '2020-03-12', start_day = '2020-09-01')
            iterv = cv.close_schools(day_schools_closed = '2020-03-12', start_day = {'pk': '2020-09-01',
            'es': 2020-09-01, 'ms': 2020-09-01, 'hs':2020-09-01, 'uv': None})
            iterv = cv.close_schools(day_schools_closed = '2020-03-12', start_day = {'pk': '2020-09-01',
            'es': 2020-09-01, 'ms': 2020-09-01, 'hs':2020-09-01, 'uv': None}. pop_file='kc_synthpops_clustered.ppl')
        '''

    def __init__(self, start_day=None, day_schools_closed=None, pop_file=None, **kwargs):
        super().__init__(**kwargs)  # Initialize the Intervention object
        self._store_args()  # Store the input arguments so that intervention can be recreated
        self.start_day          = start_day
        self.day_schools_closed = day_schools_closed
        self.pop_file           = pop_file
        self.school_types       = None
        self.contacts           = None  # list of school contacts that will need to be restored at end_day
        self.num_schools        = None
        self.new_school_network = None
        return

    def initialize(self, sim):
        if self.day_schools_closed is None:
            self.day_schools_closed = sim.day('2020-03-12')
        else:
            self.day_schools_closed = sim.day(self.day_schools_closed)

        # figure out how many schools there are and create contact list by school:
        self.num_schools = len(sim.people.schools)
        self.contacts = [None] * self.num_schools

        if isinstance(self.start_day, dict):
            self.school_types = sim.people.school_types
            for school_type, day in self.start_day.items():
                self.start_day[school_type] = sim.day(day)
        elif isinstance(self.start_day, str):
            self.start_day = sim.day(self.start_day)

        if self.pop_file is not None:
            self.load_population()

        self.initialized = True
        return

    def apply(self, sim):
        t = sim.t

        if t == self.day_schools_closed:
            for school_id in range(self.num_schools - 1): # DJK -1?
                inds_to_remove = sim.people.schools[school_id]
                self.contacts[school_id] = self.remove_contacts(inds_to_remove, 's', sim)

        else:
            if isinstance(self.start_day, dict):
                for s_type, day in self.start_day.items():
                    if day is not None:
                        if t == day:
                            schools = self.school_types[s_type]
                            if self.pop_file is not None:
                                sim.people.contacts['s'].append(self.new_school_network)
                            else:
                                for school in schools:
                                    if self.contacts[school] is not None:
                                        sim.people.contacts['s'].append(self.contacts[school])


            elif self.start_day is not None:
                if t == self.start_day:
                    if self.pop_file is not None:
                        sim.people.contacts['s'].append(self.new_school_network)
                    else:
                        for school_id in range(self.num_schools - 1): # DJK -1?
                            sim.people.contacts['s'].append(self.contacts[school_id])

    def remove_contacts(self, inds, layer, sim):
        '''Finds all contacts of a layer for a set of inds and returns an edgelist that was removed'''
        inds_list = []

        # Loop over the contact network in both directions -- k1,k2 are the keys
        for k1, k2 in [['p1', 'p2'],
                       ['p2', 'p1']]:

            # Get all the indices of the pairs that each person is in
            in_k1 = np.isin(sim.people.contacts[layer][k1], inds).nonzero()[0]
            inds_list.append(in_k1)  # Find their pairing partner
        edge_inds = np.unique(np.concatenate(inds_list))  # Find all edges
        return (sim.people.contacts[layer].pop_inds(edge_inds))

    def load_population(self):
        # Load pop_file if str else use directly for school contacts
        popfile = self.pop_file
        # Load from disk or use directly
        if isinstance(popfile, str):  # It's a string, assume it's a filename
            filepath = sc.makefilepath(filename=popfile)
            obj = cvm.load(filepath)
        else:
            obj = popfile  # Use it directly

        self.new_school_network = obj['contacts']['s']

        return


class reopen_schools(Intervention):
    '''
    Specifies school reopening strategy.
    Options:
    1. nothing
    2. screening
    3. screening with testing/tracing
    4. school closures
    5. hybrid reopening

    Args:
        ili_prev    (float or dict)     : Prevalence of influenza-like-illness symptoms in the population
        num_pos     (int)               : number of covid positive cases per school that triggers school closure
        trace       (float)             : probability of tracing contacts of diagnosed covid+
        test        (float)             : probability of testing screen positive
        test_freq   (int)               : frequency of testing teachers (1 = daily, 2 = every other day, ...)
        schedule    (bool or dict)      : whether or not to schedule partially remote (if dict, by type, otherwise for all types)
        kwargs      (dict)              : passed to Intervention

    **Examples**
        iterv = cv.reopen_schools(num_pos = 2, start_day = '2020-08-30')
        iterv = cv.reopen_schools(start_day= {'pk': None,
            'es': 2020-09-01, 'ms': 2020-09-01, 'hs':2020-09-01, 'uv': None})
    '''

    def __init__(self, start_day=None, ili_prev=None, num_pos=None, test_freq=None, trace=None, test=None,
                 schedule=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so that intervention can be recreated

        # Store arguments
        self.start_day                      = start_day
        self.ili_prev                       = ili_prev
        self.test_freq                      = test_freq # int of frequency of diagnostic testing
        self.num_pos                        = num_pos
        self.trace                          = trace # whether or not you trace contacts of all diagnosed patients
        self.test                           = test # probability that anyone who screens positive is tested
        self.schedule                       = schedule # dictionary

        # DJK TODO: Consolidate - separate structures to avoid mixing state and reporting variables
        self.date_reopen                    = None # list of dates to reopen school by
        self.contacts                       = None # list of school contacts that will need to be restored at end_day
        self.closed                         = None # whether or not each school type is closed currently
        self.school_closures                = None # total number of school closures in simulation
        self.num_schools                    = None
        self.total_days                     = None
        self.total_weeks                    = None
        self.end_day                        = None
        self.sim_end_day                    = None
        self.num_people_at_home             = None
        self.people_at_home                 = None
        self.contacts_of_people_at_home     = None
        self.next_test_day                  = None
        self.teacher_inds                   = None
        self.student_inds                   = None
        self.staff_inds                     = None
        self.school_types                   = None
        self.student_cases                  = None
        self.teacher_cases                  = None
        self.staff_cases                    = None
        self.contacts_bygroup               = None
        self.schedule_byday                 = None
        self.schools_bygroup                = None
        self.num_staff_infectious           = None
        self.num_students_infectious        = None
        self.num_diagnosed                  = None
        self.num_undiagnosed                = None
        self.beta                           = None
        self.num_students_asymptomatic      = None
        self.num_staff_asymptomatic         = None
        self.num_students                   = None
        self.school_types_open              = None
        self.school_types_closed            = None
        self.first_day                      = '2020-09-01' # DJK: Hard coded or expectation to overwrite after init?
        return

    def day_of_week(self, date):
        ''' Return the day of the week '''
        dayname = sc.readdate(date).strftime('%A')
        return dayname

    def date2group(self, date, verbose=False):
        ''' Convert a date to a day of the week and then to a policy (no_school, all, A, B, or distance)'''
        if self.schedule: # DJK: is not None?
            mapping = {
                'Monday':    'A',
                'Tuesday':   'A',
                'Wednesday': 'distance',
                'Thursday':  'B',
                'Friday':    'B',
                'Saturday':  'no_school',
                'Sunday':    'no_school',
                }
        else:
            mapping = {
                'Monday':    'all',
                'Tuesday':   'all',
                'Wednesday': 'all',
                'Thursday':  'all',
                'Friday':    'all',
                'Saturday':  'no_school',
                'Sunday':    'no_school',
                }

        if cvm.daydiff(self.first_day, date)<0: # Schools haven't opened yet
            output = 'no_school'
        else:
            dayname = self.day_of_week(date)
            output = mapping[dayname]
        if verbose:
            print(f'       debug: {date} -> {output}')
        return output


    def initialize(self, sim):
        ''' Fix the dates and create empty list for contacts '''
        self.school_types = sim.people.school_types
        self.num_schools = len(sim.people.schools)
        self.num_students = dict()
        self.school_closures = 0
        self.date_reopen = [None] * self.num_schools
        self.end_day = 14
        self.sim_end_day = sim.day(sim.pars['end_day'])
        self.total_days = self.sim_end_day - sim.day(self.first_day) + 1
        self.total_weeks = int(self.total_days/7)

        # Determine start_day
        if isinstance(self.start_day, dict):
            self.school_types = sim.people.school_types
            for school_type, day in self.start_day.items():
                self.start_day[school_type] = sim.day(day)
        elif self.start_day is not None:
            self.start_day = sim.day(self.start_day)

        # Determine which group is in school on each day
        self.schedule_byday = []
        for t in sim.tvec:
            group = self.date2group(sim.date(t))
            self.schedule_byday.append(group)

        # Prepare contacts and various reporting vars based on schedule
        if self.schedule is not None:
            self.contacts = {
                'A': [None] * self.num_schools,
                'B': [None] * self.num_schools
            }
            self.num_people_at_home = {
                'A': [0] * self.num_schools,
                'B': [0] * self.num_schools
            }
            self.people_at_home = dict()
            self.people_at_home['A'] = dict()
            self.people_at_home['B'] = dict()
            self.contacts_of_people_at_home = {
                'A': [None] * self.num_schools,
                'B': [None] * self.num_schools
            }
        else:
            self.contacts = [None] * self.num_schools
            self.num_people_at_home = [0] * self.num_schools
            self.people_at_home = dict()
            self.contacts_of_people_at_home = [None] * self.num_schools

        # More variable initializations
        self.closed = [False] * self.num_schools
        if self.trace is None:
            self.trace = 1.0
        if self.test is None:
            self.test = 0.5
        if self.test_freq is not None:
            self.next_test_day = sim.day(self.first_day)

        # Initialize the school_info dictionary # DJK: could be as separate class
        sim.school_info = dict()  # create a dict to hold information
        for var in ['num_traced', 'num_tested', 'num_teachers_tested', 'num_teachers_test_pos', 'num_teachers_screen_pos', 'num_teachers', 'num_staff_tested', 'num_staff_test_pos', 'num_staff_screen_pos', 'num_staff', 'num_students_tested', 'num_students_test_pos', 'num_students_screen_pos', 'num_students', 'school_days_lost', 'num_es', 'num_ms', 'num_hs', 'school_days_gained', 'total_student_school_days']:
            sim.school_info[var] = 0

        for var in ['num_students_infectious', 'num_staff_infectious', 'num_diagnosed', 'num_undiagnosed', 'num_staff_asymptomatic', 'num_students_asymptomatic']:
            sim.school_info[var] = []

        sim.school_info['num_students_dict'] = {}
        sim.school_info['school_days_lost_dict'] = {k:0 for k in list(self.school_types.keys()) + ['symp_quar']} # Same as above, but broken down by type
        sim.school_info['es_with_a_case'] = [0] * (self.sim_end_day + 1)
        sim.school_info['ms_with_a_case'] = [0] * (self.sim_end_day + 1)
        sim.school_info['hs_with_a_case'] = [0] * (self.sim_end_day + 1)

        ## determine which types are reopening (from cv.close_schools)
        self.types_to_check = []
        for intv in sim['interventions']:
            # Look for an intervention with label "close_school" from which to determine the school types that will need to reopen
            if intv.label == 'close_schools':
                if isinstance(intv.start_day, int):
                    self.types_to_check = ['es', 'ms', 'hs']
                elif isinstance(intv.start_day, dict):
                    self.types_to_check = []
                    for s_type, val in intv.start_day.items():
                        if val is not None:
                            if s_type != 'pk' and s_type != 'uv': # DJK: Why exclude pk and uv here?
                                self.types_to_check.append(s_type)
                else:
                    self.types_to_check = []

        # Initialization of more school_info
        for s_type in self.types_to_check:
            schools = self.school_types[s_type]
            name = 'num_' + s_type
            sim.school_info[name] = len(schools)*sim['pop_scale']
            self.count_people(schools, sim)

        all_types = ['es', 'ms', 'hs']
        for s_type in all_types:
            schools = self.school_types[s_type]
            self.count_school_days(schools, sim, s_type) # DJK confusing

        # Initialization
        sim.school_info['test_pos'] = 0
        sim.school_info['num_teacher_cases'] = 0
        sim.school_info['num_student_cases'] = 0
        sim.school_info['num_staff_cases'] = 0
        sim.school_info['num_cases_by_day'] = [0] * (self.sim_end_day + 1)
        self.teacher_inds = [i for i in range(len(sim.people.teacher_flag)) if sim.people.teacher_flag[i] is True]
        self.student_inds = [i for i in range(len(sim.people.student_flag)) if sim.people.student_flag[i] is True]
        self.staff_inds = [i for i in range(len(sim.people.staff_flag)) if sim.people.staff_flag[i] is True]

        self.student_cases = []
        self.teacher_cases = []
        self.staff_cases = []
        self.num_staff_infectious = 0
        self.num_students_infectious = 0
        self.num_diagnosed = 0
        self.num_undiagnosed = 0
        self.num_staff_asymptomatic = 0
        self.num_students_asymptomatic = 0

        self.school_types_open = []
        self.school_types_closed = []
        # count number of students lost
        # check if school is open or not
        for intv in sim['interventions']:
            if intv.label == 'close_schools':
                #N.B. start_day here is from the close_schools intervention, representing the day schools reopen
                if isinstance(intv.start_day, int):
                    self.school_types_open = ['es', 'ms', 'hs']
                elif intv.start_day is None:
                    self.school_types_closed = ['es', 'ms', 'hs']
                else:
                    for s_type, val in intv.start_day.items():
                        if val is None:
                            self.school_types_closed.append(s_type)
                        else:
                            self.school_types_open.append(s_type)

        # Hard code that we are not going to reopen pre-K or universities in this analysis
        self.school_types_closed = [stc for stc in self.school_types_closed if stc not in ['pk', 'uv']]
        self.school_types_open   = [sto for sto in self.school_types_open   if sto not in ['pk', 'uv']]

        self.initialized = True
        return


    def apply(self, sim):
        t = sim.t
        self.rescale = sim.rescale_vec[t] # Current rescaling factor for counts

        self.count_total_cases(sim)

        if t == sim.day(self.first_day):
            self.beta = sim['beta_layer']['s']
            if isinstance(self.schedule, dict):
                self.schools_bygroup = dict()
                self.schools_bygroup['A'] = dict()
                self.schools_bygroup['B'] = dict()
                self.contacts_bygroup = dict()
                self.contacts_bygroup['A'] = dict()
                self.contacts_bygroup['B'] = dict()
                for school_type, schedule in self.schedule.items():
                    if schedule is not None:
                        schools = self.school_types[school_type]
                        for school in schools:
                            students = [i for i in sim.people.schools[school] if sim.people.student_flag[i] == True]
                            teachers = [i for i in sim.people.schools[school] if sim.people.teacher_flag[i] == True]
                            staff = [i for i in sim.people.schools[school] if sim.people.staff_flag[i] == True]
                            groupA = cvu.binomial_filter(0.5, np.array(students)).tolist()
                            self.schools_bygroup['A'][school] = groupA + teachers + staff
                            groupB = [i for i in students if i not in groupA]
                            self.schools_bygroup['B'][school] = groupB + teachers + staff
                            # Remove groupA students from the 's' layer and store in contacts_bygroup['A']
                            students_to_add_groupA = self.remove_contacts(groupA, sim.people.contacts['s'])
                            self.remove_contacts(groupB, students_to_add_groupA)
                            students_that_exist_groupA = self.contacts_bygroup['A']
                            keys = ['p1', 'p2', 'beta']
                            if len(students_that_exist_groupA) > 0:
                                for key in keys:
                                    new_array = np.array(
                                        students_to_add_groupA[key].tolist() + students_that_exist_groupA[key].tolist())
                                    self.contacts_bygroup['A'][key] = new_array
                            else:
                                for key in keys:
                                    new_array = students_to_add_groupA[key]
                                    self.contacts_bygroup['A'][key] = new_array
                            students_to_add_groupB = self.remove_contacts(groupB, sim.people.contacts['s'])
                            students_that_exist_groupB = self.contacts_bygroup['B']
                            if len(students_that_exist_groupB) > 0:
                                for key in keys:
                                    new_array = np.array(
                                        students_to_add_groupB[key].tolist() + students_that_exist_groupB[key].tolist())
                                    self.contacts_bygroup['B'][key] = new_array
                            else:
                                for key in keys:
                                    new_array = students_to_add_groupB[key]
                                    self.contacts_bygroup['B'][key] = new_array

            elif self.schedule is not None:
                self.schools_groupA = dict()
                self.schools_groupB = dict()
                self.contacts_groupA = dict()
                self.contacts_groupB = dict()
                for school_id, contacts in sim.people.schools.items():
                    students = [i for i in contacts if sim.people.student_flag[i] == True]
                    teachers = [i for i in contacts if sim.people.teacher_flag[i] == True]
                    self.schools_groupA[school_id] = self.schools_groupB[school_id] = teachers
                    groupA = cvu.binomial_filter(0.5, np.array(students)).tolist()
                    self.schools_groupA[school_id].append(groupA)
                    groupB = [i for i in students if i not in groupA]
                    self.schools_groupB[school_id].append(groupB)
                    students_to_add_groupA = self.remove_contacts(groupB, 's', sim)
                    students_that_exist_groupA = self.contacts_groupA
                    keys = ['p1', 'p2', 'beta']
                    if len(students_that_exist_groupA) > 0:
                        for key in keys:
                            new_array = np.array(
                                students_to_add_groupA[key].tolist() + students_that_exist_groupA[key].tolist())
                            self.contacts_groupA[key] = new_array
                    else:
                        for key in keys:
                            new_array = students_to_add_groupA[key]
                            self.contacts_groupA[key] = new_array
                    students_to_add_groupB = self.remove_contacts(groupA, 's', sim)
                    students_that_exist_groupB = self.contacts_groupB
                    if len(students_that_exist_groupB) > 0:
                        for key in keys:
                            new_array = np.array(
                                students_to_add_groupB[key].tolist() + students_that_exist_groupB[key].tolist())
                            self.contacts_groupB[key] = new_array
                    else:
                        for key in keys:
                            new_array = students_to_add_groupB[key]
                            self.contacts_groupB[key] = new_array

        if t >= sim.day(self.first_day):
            # self.count_total_cases(sim)
            for s_type in self.types_to_check:
                schools = self.school_types[s_type]
                self.count_cases_in_schools(schools, sim)

        if isinstance(self.start_day, dict):
            self.num_students_infectious = 0
            self.num_staff_infectious = 0
            self.num_diagnosed = 0
            self.num_undiagnosed = 0
            self.num_staff_asymptomatic = 0
            self.num_students_asymptomatic = 0
            for s_type, day in self.start_day.items():
                if day is not None:
                    if t >= day:
                        group = self.schedule_byday[t]
                        if group == 'no_school':
                            # weekend
                            sim['beta_layer']['s'] = 0
                        elif group == 'distance':
                            # day at home
                            sim['beta_layer']['s'] = 0
                            lost = sim['pop_scale'] * self.num_students[s_type]
                            sim.school_info['school_days_lost'] += lost
                            sim.school_info['school_days_lost_dict'][s_type] += lost
                        else:
                            sim['beta_layer']['s'] = self.beta
                            if group == 'all':
                                sim.school_info['school_days_gained'] += sim['pop_scale']*self.num_students[s_type]
                            else:
                                # if Hybrid (A or B), then count half the students as being home
                                lost = sim['pop_scale']*self.num_students[s_type]/2
                                sim.school_info['school_days_lost'] += lost
                                sim.school_info['school_days_lost_dict'][s_type] += lost
                                sim.school_info['school_days_gained'] += lost # Gained is same as lost
                                for key in sim.people.contacts['s'].keys():
                                    sim.people.contacts['s'][key] = self.contacts_bygroup[group][key] # Load in group contacts

                            for i in self.school_types[s_type]:
                                if self.closed[i]:
                                    if t == self.date_reopen[i]:
                                        self.reopen_school(sim, i)
                                    else:
                                        self.check_on_people_at_home(sim, i)
                                else:
                                    self.check_condition(sim, i, s_type, group)
                                    if group == 'all':
                                        #all in person
                                        self.count_school_days_lost(i, sim, group)
                                        if self.num_people_at_home[i] > 0:
                                            self.check_on_people_at_home(sim, i, group)
                                            if t in self.contacts_of_people_at_home[i]['timer']:
                                                self.return_to_school(sim, i, group)
                                    else:
                                        self.count_school_days_lost(i, sim, group)
                                        # either A or B day
                                        if self.num_people_at_home[group][i] > 0:
                                            self.check_on_people_at_home(sim, i, group)
                                            if len(self.contacts_of_people_at_home[group][i]['timer']) > 0:
                                                min_time = min(self.contacts_of_people_at_home[group][i]['timer'])
                                                if t >= min_time:
                                                    self.return_to_school(sim, i, group)

                        # DJK: what is this doing? Replacing contacts_bygroup with contacts?
                        # Oh, above we loaded the appropriate group into contacts, now "returning" them to contacts_bygroup
                        if group in ['A', 'B']:
                            for key in sim.people.contacts['s'].keys():
                                self.contacts_bygroup[group][key] = sim.people.contacts['s'][key]

                else:
                    if t >= sim.day(self.first_day): # t must be before (<) the start day for this school type:
                        group = self.schedule_byday[t]
                        if group != 'no_school':
                            if s_type in self.school_types_closed and self.num_students[s_type]:
                                # school never opened, so all students are home
                                lost = sim['pop_scale']*self.num_students[s_type]
                                sim.school_info['school_days_lost'] += lost
                                sim.school_info['school_days_lost_dict'][s_type] += lost

            if self.test_freq is not None:
                # If testing today, schedule the next_test_day by adding the routine testing frequency
                # Routine testing is accomplished in check_condition, which is called above
                # DJK: TODO - make test_freq and associated variables a dictionary by type (students, teachers, staff) to have dependent frequencies
                if t == self.next_test_day and t < (self.sim_end_day - 2):
                    self.next_test_day += self.test_freq
                    if self.next_test_day <= len(self.schedule_byday):
                        if not self.schedule_byday[self.next_test_day]: # DJK: schedule_byday appears to be consistently defined, so can remove and remove +2 here and above
                            self.next_test_day += 2

            sim.school_info['num_staff_infectious'].append(self.num_staff_infectious)
            sim.school_info['num_students_infectious'].append(self.num_students_infectious)
            sim.school_info['num_diagnosed'].append(self.num_diagnosed)
            sim.school_info['num_staff_asymptomatic'].append(self.num_staff_asymptomatic)
            sim.school_info['num_students_asymptomatic'].append(self.num_students_asymptomatic)
        elif self.start_day is not None:
            if t >= self.start_day:
                for i in range(self.num_schools - 1):
                    if self.closed[i]:
                        if t == self.date_reopen[i]:
                            self.reopen_school(sim, i)
                        else:
                            self.check_on_people_at_home(sim, i)
                    else:
                        self.check_condition(sim, i, None)
                        if self.num_people_at_home[i] > 0:
                            self.check_on_people_at_home(sim, i)
                            if t in self.contacts_of_people_at_home[i]['timer']:
                                self.return_to_school(sim, i)
                if self.next_test_day is not None:
                    self.next_test_day += self.test_freq
        else:
            if t >= sim.day(self.first_day):
                group = self.schedule_byday[t]
                if group != 'no_school':
                    # If all schools are closed, count non-weekend days
                    for s_type in self.school_types_closed:
                        lost = sim['pop_scale']*self.num_students[s_type]
                        sim.school_info['school_days_lost'] += lost
                        sim.school_info['school_days_lost_dict'][s_type] += lost
                    #If school is as normal (no intervention)
                    for s_type in self.school_types_open:
                        sim.school_info['school_days_gained'] += sim['pop_scale']*self.num_students[s_type]
                        schools = self.school_types[s_type]
                        self.count_school_days_lost(schools, sim)
                        self.count_infections_in_school(schools, s_type, sim)

        if t == self.sim_end_day:
            sim.school_info['school_closures'] = self.school_closures
            if self.teacher_cases is not None:
                self.teacher_cases = list(dict.fromkeys(self.teacher_cases))
                sim.school_info['num_teacher_cases'] = self.rescale*len(self.teacher_cases) # WARNING, these are not correctly scaled!
            if self.student_cases is not None:
                self.student_cases = list(dict.fromkeys(self.student_cases))
                sim.school_info['num_student_cases'] = self.rescale*len(self.student_cases)
            if self.staff_cases is not None:
                self.staff_cases = list(dict.fromkeys(self.staff_cases))
                sim.school_info['num_staff_cases'] = self.rescale*len(self.staff_cases)
        return

    def count_school_days_lost(self, schools, sim, group=None):
        #potentially call this function only for schools in session
        list_of_students_in_school = []
        if isinstance(schools, list):
            for school in schools:
                students_in_school = sim.people.schools[school]
                list_of_students_in_school.append(students_in_school)
        else:
            if group in ['A', 'B']:
                students_in_school = self.schools_bygroup[group][schools]
                list_of_students_in_school = [students_in_school]
            elif group == 'all':
                students_in_school = sim.people.schools[schools]
                list_of_students_in_school = [students_in_school]

        if len(list_of_students_in_school) >0:
            for students_in_school in list_of_students_in_school:
                inds_quarantined = np.array(students_in_school)[
                    sim.people.quarantined[np.array(students_in_school)]].tolist()
                inds_symptomatic = np.array(students_in_school)[
                    sim.people.symptomatic[np.array(students_in_school)]].tolist()
                inds_symptomatic = [x for x in inds_symptomatic if x not in inds_quarantined]
                symp_lost = self.rescale * len(inds_symptomatic)
                quar_lost = self.rescale * len(inds_quarantined)
                lost_total = symp_lost + quar_lost
                sim.school_info['school_days_lost'] += lost_total
                sim.school_info['school_days_lost_dict']['symp_quar'] += lost_total
                sim.school_info['school_days_gained'] -= lost_total # Not otherwise counted

        return

    def count_cases_in_schools(self, schools, sim):
        for school in schools:
            students_in_school = sim.people.schools[school]
            school_infectious = cvu.itrue(sim.people.infectious[np.array(students_in_school)], np.array(students_in_school))
            students_infectious = [x for x in school_infectious.tolist() if x in self.student_inds]
            staff_infectious = [x for x in school_infectious.tolist() if x in self.staff_inds]
            teacher_infectious = [x for x in school_infectious.tolist() if x in self.teacher_inds]
            if len(students_infectious) > 0:
                self.student_cases += students_infectious # CK: WARNING, these are not scaled correctly
            if len(teacher_infectious) > 0:
                self.teacher_cases += teacher_infectious
            if len(staff_infectious) > 0:
                self.staff_cases += staff_infectious
        return

    def count_total_cases(self, sim):
        students_infectious = cvu.itrue(sim.people.infectious[np.array(self.student_inds)], np.array(self.student_inds))
        staff_infectious = cvu.itrue(sim.people.infectious[np.array(self.staff_inds)], np.array(self.staff_inds))
        teachers_infectious = cvu.itrue(sim.people.infectious[np.array(self.teacher_inds)], np.array(self.teacher_inds))
        sim.school_info['num_cases_by_day'][sim.t] += self.rescale*(len(students_infectious) + len(staff_infectious) + len(teachers_infectious))
        return

    def count_school_days(self, schools, sim, s_type):
        school_days = len(self.schedule_byday) - self.schedule_byday.count('no_school')
        # school_days = sim.day(sim.pars['end_day']) - sim.day(self.first_day)
        # num_weeks = school_days/7
        # weekend_days = 2*num_weeks
        # school_days = school_days - weekend_days
        num_students = 0
        num_staff = 0
        num_teachers = 0
        for school in schools:
            individuals_in_school = sim.people.schools[school]
            for indiv in individuals_in_school:
                if sim.people.student_flag[indiv]:
                    num_students += 1
                elif sim.people.teacher_flag[indiv]:
                    num_teachers += 1
                elif sim.people.staff_flag[indiv]:
                    num_staff += 1
        self.num_students[s_type] = num_students
        sim.school_info['num_students_dict'][s_type] = num_students
        sim.school_info['total_student_school_days'] += sim['pop_scale']*school_days*num_students # This is total, not by infected students

    def count_people(self, schools, sim):
        num_students = 0
        num_staff = 0
        num_teachers = 0
        for school in schools:
            individuals_in_school = sim.people.schools[school]
            for indiv in individuals_in_school:
                if sim.people.student_flag[indiv]:
                    num_students += 1
                elif sim.people.teacher_flag[indiv]:
                    num_teachers += 1
                elif sim.people.staff_flag[indiv]:
                    num_staff += 1
        sim.school_info['num_students'] += sim['pop_scale']*num_students
        sim.school_info['num_teachers'] += sim['pop_scale']*num_teachers
        sim.school_info['num_staff'] += sim['pop_scale']*num_staff

    def count_infections_in_school(self, schools, school_type, sim):
        t = sim.t
        for school_id in schools:
            school = sim.people.schools[school_id]
            school_infectious = cvu.itrue(sim.people.infectious[np.array(school)], np.array(school))
            if len(school_infectious) > 0:
                name = school_type + '_with_a_case'
                sim.school_info[name][t] += self.rescale*1 # CK: left the *1 as we used to count single schools
        return

    def screen(self, sim, school):
        inds_meet_condition = []
        if sim.pars['pop_type'] == 'hybrid':
            for classroom in school:
                inds_meet_condition += np.array(classroom)[sim.people.diagnosed[np.array(classroom)]].tolist()
                inds_meet_condition += np.array(classroom)[sim.people.symptomatic[np.array(classroom)]].tolist()
                if self.ili_prev is not None:
                    n_ili = int(self.ili_prev * len(classroom))  # Number with ILI symptoms on this day
                    inds_meet_condition += np.random.choice(classroom, n_ili, replace=False).tolist()
                inds_meet_condition = np.array(inds_meet_condition)[~sim.people.recovered[inds_meet_condition]].tolist()
                inds_meet_condition = np.array(inds_meet_condition)[~sim.people.dead[inds_meet_condition]].tolist()
                inds_meet_condition = list(dict.fromkeys(inds_meet_condition))
        else:
            inds_meet_condition += np.array(school)[sim.people.diagnosed[np.array(school)]].tolist()
            inds_meet_condition += np.array(school)[sim.people.symptomatic[np.array(school)]].tolist()
            inds_meet_condition = np.array(inds_meet_condition)[~sim.people.recovered[inds_meet_condition]].tolist()
            inds_meet_condition = np.array(inds_meet_condition)[~sim.people.dead[inds_meet_condition]].tolist()
            if self.ili_prev is not None:
                n_ili = int(self.ili_prev * len(school))  # Number with ILI symptoms on this day
                inds_meet_condition += np.random.choice(school, n_ili, replace=False).tolist()
            inds_meet_condition = list(dict.fromkeys(inds_meet_condition))
            students_screen_pos = [x for x in inds_meet_condition if x in self.student_inds]
            teachers_screen_pos = [x for x in inds_meet_condition if x in self.teacher_inds]
            staff_screen_pos = [x for x in inds_meet_condition if x in self.staff_inds]
            sim.school_info['num_teachers_screen_pos'] += len(teachers_screen_pos)
            sim.school_info['num_students_screen_pos'] += len(students_screen_pos)
            sim.school_info['num_staff_screen_pos'] += len(staff_screen_pos)
        return inds_meet_condition

    def trace_contacts(self, inds, sim, school_id, group):
        # trace contacts of diagnosed person and remove their student contacts from school for 14 days
        sim.people.trace(inds, trace_probs={'h': 1, 'c': 0, 'w': 0, 's': 1},
                         trace_time={'h': 2, 'c': 2, 'w': 2, 's': 2})

        # DJK: What about the delay?
        # DJK: Can just use .quarantined?
        # DJK: Only works with trace_probs of 1?
        for student in inds:
            contacts_list = self.remove_contacts(student, sim.people.contacts['s'])
            contacts = contacts_list['p1']
            contacts = np.append(contacts, contacts_list['p2'])
            contacts = np.unique(contacts).tolist()
            contacts = [x for x in contacts if x not in [student]]
            if len(contacts)>0:
                sim.school_info['num_traced'] += self.rescale*len(contacts)
                sim.people.test(contacts, test_delay=0)
                sim.school_info['num_tested'] += self.rescale*len(contacts)
                teachers_tested = [x for x in contacts if x in self.teacher_inds]
                staff_tested = [x for x in contacts if x in self.staff_inds]
                students_tested = [x for x in contacts if x in self.student_inds]
                sim.school_info['num_students_tested'] += self.rescale*len(students_tested)
                sim.school_info['num_teachers_tested'] += self.rescale*len(teachers_tested)
                sim.school_info['num_staff_tested'] += self.rescale*len(staff_tested)
                inds_covid = cvu.itrue(sim.people.infectious[np.array(contacts)], np.array(contacts)).tolist()
                sim.school_info['test_pos'] += self.rescale*len(inds_covid)
                if len(inds_covid) > 0:
                    teachers_pos = [x for x in inds_covid if x in self.teacher_inds]
                    staff_pos = [x for x in inds_covid if x in self.staff_inds]
                    student_pos = [x for x in inds_covid if x in self.student_inds]
                    sim.school_info['num_students_test_pos'] += self.rescale*len(student_pos)
                    sim.school_info['num_teachers_test_pos'] += self.rescale*len(teachers_pos)
                    sim.school_info['num_staff_test_pos'] += self.rescale*len(staff_pos)
                self.remove_from_school(sim, school_id, contacts, group)
        return

    def check_condition(self, sim, school_id, school_type, group=None):
        '''check if school meets closure condition

        1. Count number of covid positive cases in school
        2. Test screen positives
        3. If > self.num_pos, close school
        4. If not, send screen positives home
        5. If it's a testing day, test teachers
        '''

        t = sim.t

        # retrieve inds of students/teachers in school
        if group in ['A', 'B']:
            school = self.schools_bygroup[group][school_id]
        else:
            school = sim.people.schools[school_id]
        num_covid_pos = 0

        # count number of infectious students, teachers and staff
        school_infectious = cvu.itrue(sim.people.infectious[np.array(school)], np.array(school))
        school_asymptomatic = cvu.ifalse(sim.people.symptomatic[school_infectious], school_infectious)
        students_infectious = [x for x in school_infectious.tolist() if x in self.student_inds]
        staff_infectious = [x for x in school_infectious.tolist() if x in self.staff_inds]
        teacher_infectious = [x for x in school_infectious.tolist() if x in self.teacher_inds]
        students_asymptomatic = [x for x in school_asymptomatic.tolist() if x in self.student_inds]
        staff_asymptomatic = [x for x in school_asymptomatic.tolist() if x in self.staff_inds or x in self.teacher_inds]
        school_diagnosed = cvu.itrue(sim.people.diagnosed[np.array(school)], np.array(school))
        school_diagnosed = np.array(school_diagnosed)[~sim.people.recovered[school_diagnosed]].tolist()
        school_diagnosed = np.array(school_diagnosed)[~sim.people.dead[school_diagnosed]].tolist()
        school_undiagnosed = [x for x in school_infectious if x not in school_diagnosed]
        self.num_students_infectious += self.rescale*len(students_infectious)
        self.num_staff_infectious += self.rescale*len(staff_infectious)
        self.num_staff_infectious += self.rescale*len(teacher_infectious)
        self.num_diagnosed += self.rescale*len(school_diagnosed)
        self.num_undiagnosed += self.rescale*len(school_undiagnosed)
        self.num_staff_asymptomatic += self.rescale*len(staff_asymptomatic)
        self.num_students_asymptomatic += self.rescale*len(students_asymptomatic)

        # check if anyone is currently at home, count how many are Covid+
        # screen everyone who is in school
        teachers_tested = []
        staff_tested = []
        inds_meet_condition = []
        if group in ['A', 'B']:
            already_home = self.num_people_at_home[group][school_id]
        else:
            already_home = self.num_people_at_home[school_id]

        if already_home > 0:
            if group in ['A', 'B']:
                people_at_home = self.people_at_home[group][school_id]
            else:
                people_at_home = self.people_at_home[school_id]

            students_with_diagnosed_covid_at_home = cvu.itrue(sim.people.diagnosed[np.array(people_at_home)], np.array(people_at_home))
            students_with_diagnosed_covid_at_home = np.array(students_with_diagnosed_covid_at_home)[~sim.people.recovered[students_with_diagnosed_covid_at_home]].tolist()
            students_with_diagnosed_covid_at_home = np.array(students_with_diagnosed_covid_at_home)[~sim.people.dead[students_with_diagnosed_covid_at_home]].tolist()
            num_covid_pos += self.rescale*len(students_with_diagnosed_covid_at_home)
            if isinstance(people_at_home, int):
                people_at_home = [people_at_home]
            infectious_in_school = [x for x in school_infectious.tolist() if x not in people_at_home]
            inds_in_school = [x for x in school if x not in people_at_home]
            if len(inds_in_school)>0:
                inds_meet_condition = self.screen(sim, inds_in_school)
        else:
            inds_meet_condition = self.screen(sim, school)
            infectious_in_school = school_infectious.tolist()

        if len(infectious_in_school)>0:
            name = school_type + '_with_a_case'
            sim.school_info[name][t] += self.rescale*1 # CK: non-integer schools might cause confusion, but is correct I think!

        # choose which people get tested
        if len(inds_meet_condition)>0:
            tested = cvu.n_binomial(self.test, len(inds_meet_condition))
            inds_to_test = np.array(inds_meet_condition)[tested]
            if len(inds_to_test) > 0:
                sim.people.test(inds_to_test, test_delay=0)
                sim.school_info['num_tested'] += self.rescale*len(inds_to_test)
                teachers_tested = [x for x in inds_to_test if x in self.teacher_inds]
                staff_tested = [x for x in inds_to_test if x in self.staff_inds]
                students_tested = [x for x in inds_to_test if x in self.student_inds]
                sim.school_info['num_students_tested'] += self.rescale*len(students_tested)
                sim.school_info['num_teachers_tested'] += self.rescale*len(teachers_tested)
                sim.school_info['num_staff_tested'] += self.rescale*len(staff_tested)
                # choose which contacts get traced
                inds_covid = cvu.itrue(sim.people.infectious[inds_to_test], inds_to_test) # TODO: Test sensitivity here?
                num_covid_pos += self.rescale*len(inds_covid)
                sim.school_info['test_pos'] += self.rescale*len(inds_covid)
                if len(inds_covid) > 0:
                    teachers_pos = [x for x in inds_covid if x in self.teacher_inds]
                    staff_pos = [x for x in inds_covid if x in self.staff_inds]
                    student_pos = [x for x in inds_covid if x in self.student_inds]
                    sim.school_info['num_students_test_pos'] += self.rescale*len(student_pos)
                    sim.school_info['num_teachers_test_pos'] += self.rescale*len(teachers_pos)
                    sim.school_info['num_staff_test_pos'] += self.rescale*len(staff_pos)
                    traced = cvu.n_binomial(self.trace, len(inds_covid))
                    inds_to_trace = np.array(inds_covid)[traced]
                    if len(inds_to_trace) > 0:
                        self.trace_contacts(inds_to_trace, sim, school_id, group)

        # determine if we are doing any routine diagnostic testing among teachers & staff today
        if self.next_test_day is not None and t == self.next_test_day:
            # Time for routine testing
            if already_home > 0:
                if group in ['A', 'B']:
                    people_at_home = self.people_at_home[group][school_id]
                else:
                    people_at_home = self.people_at_home[school_id]
                if isinstance(people_at_home, int):
                    people_at_home = [people_at_home]
            else:
                people_at_home = []

            in_school = [x for x in school if x not in people_at_home]

            # Routine testing for teachers
            teacher_inds_in_school = [x for x in in_school if x in self.teacher_inds]
            if len(teacher_inds_in_school)>0:
                asymp_teacher_inds = [x for x in teacher_inds_in_school if x not in inds_meet_condition]
                if len(asymp_teacher_inds) > 0:
                    # Test teachers who are in school who don't meet screening conditions (diagnosed, symptomatic, ~recovered, ~dead)
                    sim.people.test(asymp_teacher_inds, test_delay=0)
                    sim.school_info['num_teachers_tested'] += self.rescale*len(asymp_teacher_inds)
                    sim.school_info['num_tested'] += self.rescale*len(asymp_teacher_inds)
                    teacher_inds_covid = cvu.itrue(sim.people.infectious[np.array(asymp_teacher_inds)], np.array(asymp_teacher_inds))
                    print(f'Teacher testing: {self.rescale*len(teacher_inds_covid)} pos of {self.rescale*len(asymp_teacher_inds)}')
                    num_covid_pos += self.rescale*len(teacher_inds_covid)
                    if len(teacher_inds_covid) > 0:
                        sim.school_info['num_teachers_test_pos'] += self.rescale*len(teacher_inds_covid)
                        traced = cvu.n_binomial(self.trace, len(teacher_inds_covid))
                        inds_to_trace = teacher_inds_covid[traced]
                        if len(inds_to_trace) > 0:
                            self.trace_contacts(inds_to_trace, sim, school_id, group)
                        not_in_inds_meet_condition = [x for x in teacher_inds_covid.tolist() if
                                                      x not in inds_meet_condition]
                        inds_meet_condition += not_in_inds_meet_condition

            # Routine testing for staff
            staff_inds_in_school = [x for x in in_school if x in self.staff_inds]
            if len(staff_inds_in_school)>0:
                asymp_staff_inds = [x for x in staff_inds_in_school if x not in inds_meet_condition]
                if len(asymp_staff_inds) > 0:
                    # Test staff who are in school who don't meet screening conditions (diagnosed, symptomatic, ~recovered, ~dead)
                    sim.people.test(asymp_staff_inds, test_delay=0)
                    staff_inds_covid = cvu.itrue(sim.people.infectious[np.array(asymp_staff_inds)], np.array(asymp_staff_inds))
                    print(f'Staff testing: {self.rescale*len(staff_inds_covid)} pos of {self.rescale*len(asymp_staff_inds)}')
                    num_covid_pos += self.rescale*len(staff_inds_covid)
                    sim.school_info['test_pos'] += self.rescale*len(staff_inds_covid)
                    sim.school_info['num_staff_tested'] += self.rescale*len(asymp_staff_inds)
                    sim.school_info['num_tested'] += self.rescale*len(asymp_staff_inds)
                    if len(staff_inds_covid) > 0:
                        sim.school_info['num_staff_test_pos'] += self.rescale*len(staff_inds_covid)
                        traced = cvu.n_binomial(self.trace, len(staff_inds_covid))
                        inds_to_trace = staff_inds_covid[traced]
                        if len(inds_to_trace) > 0:
                            self.trace_contacts(inds_to_trace, sim, school_id, group)
                        not_in_inds_meet_condition = [x for x in staff_inds_covid.tolist() if
                                                      x not in inds_meet_condition]
                        inds_meet_condition += not_in_inds_meet_condition

        if self.num_pos is not None:
            if num_covid_pos >= self.num_pos:
                self.close_school(sim, school_id)
            elif len(inds_meet_condition) > 0:
                self.remove_from_school(sim, school_id, inds_meet_condition, group)
        elif len(inds_meet_condition) > 0:
            self.remove_from_school(sim, school_id, inds_meet_condition, group)

        return

    def close_school(self, sim, school_id):
        ''' Closes a school, removes contacts, and sets reopen date. IF there are any students from school
        currently at home, adds their contacts to school for future use and resets people_at_home'''

        # retrieve edge-list of anyone currently at home
        contacts_of_people_at_home = self.contacts_of_people_at_home[school_id]

        # list of students to send home
        school = sim.people.schools[school_id]
        inds_to_remove = []
        if sim.pars['pop_type'] == 'hybrid':
            for classroom in school:
                inds_to_remove += classroom
        else:
            inds_to_remove = school
        if contacts_of_people_at_home is not None:
            people_at_home = self.people_at_home[school_id]
            if isinstance(people_at_home, int):
                people_at_home = [people_at_home]
            if isinstance(inds_to_remove, int):
                inds_to_remove = [inds_to_remove]
            inds_to_remove = [x for x in inds_to_remove if x not in people_at_home]
            students_to_remove = self.remove_contacts(inds_to_remove, sim.people.contacts['s'])
            self.contacts[school_id] = {}
            keys = ['p1', 'p2', 'beta']
            for key in keys:
                new_array = np.array(contacts_of_people_at_home[key].tolist() + students_to_remove[key].tolist())
                self.contacts[school_id][key] = new_array
            self.contacts_of_people_at_home[school_id] = None
            self.num_people_at_home[school_id] = 0
        else:
            self.contacts[school_id] = self.remove_contacts(inds_to_remove, sim.people.contacts['s'])
        self.closed[school_id] = True
        self.school_closures += 1

        # student_inds = [x for x in inds_to_remove if x in self.student_inds]
        # sim.school_info['school_days_lost'] += (self.end_day * len(student_inds))
        self.date_reopen[school_id] = sim.t + self.end_day
        return

    def reopen_school(self, sim, school_id):
        ''' Reopens a school, adds contacts back.'''
        sim.people.contacts['s'].append(self.contacts[school_id])
        self.closed[school_id] = False
        return

    def remove_contacts(self, inds, contacts):
        '''Finds all contacts of a layer for a set of inds and returns an edgelist that was removed'''
        inds_list = []
        for k1, k2 in [['p1', 'p2'],
                       ['p2', 'p1']]:  # Loop over the contact network in both directions -- k1,k2 are the keys
            in_k1 = np.isin(contacts[k1], inds).nonzero()[
                0]  # Get all the indices of the pairs that each person is in
            inds_list.append(in_k1)  # Find their pairing partner
        edge_inds = np.unique(np.concatenate(inds_list)) # Find all edges

        output = {}
        for key, value in contacts.items():
            output[key] = value[edge_inds]  # Copy to the output object
            contacts[key] = np.delete(value, edge_inds)  # Remove from the original
        return output

    def remove_from_school(self, sim, school_id, inds, group=None):

        if group == 'A' or group == 'B':
            contacts_at_home = self.contacts_of_people_at_home[group][school_id]
            num_people_at_home = self.num_people_at_home[group][school_id]
        else:
            contacts_at_home = self.contacts_of_people_at_home[school_id]
            num_people_at_home = self.num_people_at_home[school_id]

        if contacts_at_home is not None:
            if group == 'A' or group == 'B':
                people_at_home = self.people_at_home[group][school_id]
            else:
                people_at_home = self.people_at_home[school_id]
            students_to_add = self.remove_contacts(inds, sim.people.contacts['s'])
            students_to_add['timer'] = np.array([sim.t + self.end_day] * len(students_to_add['p1']))
            students_that_exist = contacts_at_home
            keys = ['p1', 'p2', 'beta', 'timer']
            for key in keys:
                new_array = np.array(students_to_add[key].tolist() + students_that_exist[key].tolist())
                contacts_at_home[key] = new_array
            if isinstance(inds, int):
                inds = [inds]
            if isinstance(people_at_home, int):
                people_at_home = [people_at_home]
            people_at_home = people_at_home + inds
        else:
            contacts_at_home = self.remove_contacts(inds, sim.people.contacts['s'])
            contacts_at_home['timer'] = np.array([sim.t + self.end_day] * len(contacts_at_home['p1']))
            if isinstance(inds, int):
                inds = [inds]
            people_at_home = inds
        num_people_at_home += len(inds)
        # student_inds = [x for x in inds if x in self.student_inds]
        # sim.school_info['school_days_lost'] += (self.end_day * len(student_inds))

        # update contacts, inds, and number
        if group == 'A' or group == 'B':
            self.contacts_of_people_at_home[group][school_id] = contacts_at_home
            self.people_at_home[group][school_id] = people_at_home
            self.num_people_at_home[group][school_id] = num_people_at_home
        else:
            self.contacts_of_people_at_home[school_id] = contacts_at_home
            self.people_at_home[school_id] = people_at_home
            self.num_people_at_home[school_id] = num_people_at_home
        return

    def return_to_school(self, sim, school_id, group=None):
        # retrieve dictionary of those whose timer is up
        if group == 'A' or group == 'B':
            contacts_of_all_people_at_home = self.contacts_of_people_at_home[group][school_id]
            timers = contacts_of_all_people_at_home['timer']
            inds = sc.findinds(timers <= sim.t)
            contacts_to_return = {}
            keys = ['p1', 'p2', 'beta']
            for key in keys:
                contacts_to_return[key] = contacts_of_all_people_at_home[key][inds]
                self.contacts_of_people_at_home[group][school_id][key] = np.delete(
                    self.contacts_of_people_at_home[group][school_id][key], [inds])

            self.contacts_of_people_at_home[group][school_id]['timer'] = np.delete(
                self.contacts_of_people_at_home[group][school_id]['timer'],
                [inds])

            sim.people.contacts['s'].append(contacts_to_return)
            contacts_returned = contacts_to_return['p1']
            contacts_returned = np.append(contacts_returned, contacts_to_return['p2'])
            contacts_returned = np.unique(contacts_returned)
            if isinstance(self.people_at_home[group][school_id], int):
                students_returned = [self.people_at_home[group][school_id]]
            else:
                students_returned = set(contacts_returned).intersection(set(self.people_at_home[group][school_id]))
                students_returned = list(students_returned)
            self.num_people_at_home[group][school_id] -= len(students_returned)
            if isinstance(self.people_at_home[group][school_id], int):
                self.people_at_home[group][school_id] = [self.people_at_home[group][school_id]]
            self.people_at_home[group][school_id] = self.people_at_home[group][school_id][
                self.people_at_home[group][school_id] != students_returned]
        else:
            contacts_of_all_people_at_home = self.contacts_of_people_at_home[school_id]
            timers = contacts_of_all_people_at_home['timer']
            inds = sc.findinds(timers <= sim.t)
            contacts_to_return = {}
            keys = ['p1', 'p2', 'beta']
            for key in keys:
                contacts_to_return[key] = contacts_of_all_people_at_home[key][inds]
                self.contacts_of_people_at_home[school_id][key] = np.delete(
                    self.contacts_of_people_at_home[school_id][key], [inds])

            self.contacts_of_people_at_home[school_id]['timer'] = np.delete(
                self.contacts_of_people_at_home[school_id]['timer'],
                [inds])

            sim.people.contacts['s'].append(contacts_to_return)
            contacts_returned = contacts_to_return['p1']
            contacts_returned = np.append(contacts_returned, contacts_to_return['p2'])
            contacts_returned = np.unique(contacts_returned)
            if isinstance(self.people_at_home[school_id], int):
                students_returned = [self.people_at_home[school_id]]
            else:
                students_returned = set(contacts_returned).intersection(set(self.people_at_home[school_id]))
                students_returned = list(students_returned)
            self.num_people_at_home[school_id] -= len(students_returned)
            if isinstance(self.people_at_home[school_id], int):
                self.people_at_home[school_id] = [self.people_at_home[school_id]]
            self.people_at_home[school_id] = self.people_at_home[school_id][
                self.people_at_home[school_id] != students_returned]
        return

    def check_on_people_at_home(self, sim, school_id, group=None):
        if self.closed[school_id]:
            students = sim.people.schools[school_id]
        else:
            if group == 'A' or group == 'B':
                students = self.people_at_home[group][school_id]
            else:
                students = self.people_at_home[school_id]
            if isinstance(students, int):
                students = [students]

        # Update timers of those who tested negative
        inds_tested = cvu.itrue(sim.people.tested[np.array(students)], np.array(students))
        if len(inds_tested)>0:
            inds_no_covid = cvu.ifalse(sim.people.diagnosed[inds_tested], inds_tested)
            if len(inds_no_covid) > 0:
                if not self.closed[school_id]:
                    inds_not_quarantined = cvu.ifalse(sim.people.quarantined[inds_no_covid], inds_no_covid)
                    self.update_timers(school_id, inds_not_quarantined, sim, group)
        return

    def update_timers(self, school_id, inds, sim, group=None):
        if group == 'A' or group == 'B':
            contacts_of_all_people_at_home = self.contacts_of_people_at_home[group][school_id]
        else:
            contacts_of_all_people_at_home = self.contacts_of_people_at_home[school_id]
        for student in inds:
            inds_to_update = sc.findinds(contacts_of_all_people_at_home['p1'] == student)
            inds_to_update = np.append(inds_to_update, sc.findinds(contacts_of_all_people_at_home['p2'] == student))
            inds_to_update = np.unique(inds_to_update)
            contacts_of_student = sc.findinds(contacts_of_all_people_at_home['p1'] == student)
            contacts_of_student = np.append(contacts_of_student, sc.findinds(contacts_of_all_people_at_home['p2'] == student))
            # if len(contacts_of_student) > 0:
                # contact = contacts_of_student[0]
                # timer = contacts_of_all_people_at_home['timer'][contact]
                # sim.school_info['school_days_lost'] -= (timer - sim.t + 3)
            if group == 'A' or group == 'B':
                self.contacts_of_people_at_home[group][school_id]['timer'][inds_to_update] = \
                    np.repeat(3+sim.t, len(inds_to_update))
            else:
                self.contacts_of_people_at_home[school_id]['timer'][inds_to_update] = np.repeat(3 + sim.t,
                                                                                                  len(inds_to_update))

