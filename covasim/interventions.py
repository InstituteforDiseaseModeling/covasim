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
from . import base as cvb



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
            if key not in ['self', '__class__']: # Skip these two
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
        Apply intervention

        Function signature matches existing intervention definition
        This method gets called at each timestep and must be implemented
        by derived classes

        Args:
            sim: The Sim instance

        Returns:
            None
        '''
        raise NotImplementedError


    def plot(self, sim, ax=None, **kwargs):
        '''
        Call function during plotting

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
        attributes, then its `to_json` method will need to handle those

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
        self.initialized = True
        return


    def apply(self, sim):
        for ind in find_day(self.days_arr <= sim.t, which='last'):
            self.interventions[ind].apply(sim)
        return



#%% Beta interventions

__all__+= ['change_beta', 'clip_edges']


def process_days(sim, days):
    '''
    Ensure lists of days are in consistent format. Used by change_beta, clip_edges,
    and some analyzers. If day is 'end' or -1, use the final day of the simulation.
    '''
    if sc.isstring(days) or not sc.isiterable(days):
        days = sc.promotetolist(days)
    if isinstance(days, list):
        for d,day in enumerate(days):
            if day in ['end', -1]:
                day = sim['end_day']
            days[d] = sim.day(day) # Ensure it's an integer and not a string or something
    days = sc.promotetoarray(days)
    return days


def process_changes(sim, changes, days):
    '''
    Ensure lists of changes are in consistent format. Used by change_beta and clip_edges.
    '''
    changes = sc.promotetoarray(changes)
    if len(days) != len(changes):
        errormsg = f'Number of days supplied ({len(days)}) does not match number of changes in beta ({len(changes)})'
        raise ValueError(errormsg)
    return changes


class change_beta(Intervention):
    '''
    The most basic intervention -- change beta by a certain amount.

    Args:
        days (int or array): the day or array of days to apply the interventions
        changes (float or array): the changes in beta (1 = no change, 0 = no transmission)
        layers (str or list): the layers in which to change beta
        kwargs (dict): passed to Intervention()

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

        # Ensure the edges get deleted at the end
        if sim.t == sim.tvec[-1]:
            self.contacts = None # Reset to save memory

        return



#%% Testing interventions

__all__+= ['test_num', 'test_prob', 'contact_tracing']


# Process daily data
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
    to other interventions.

    Args:
        subtarget (dict): dict with keys 'inds' and 'vals'; see test_num() for examples of a valid subtarget dictionary
        sim (Sim): the simulation object
    '''

    # Validation
    if 'inds' not in subtarget:
        errormsg = f'The subtarget dict must have keys "inds" and "vals", but you supplied {subtarget}'
        raise ValueError(errormsg)

    # Handle the two options of type
    if callable(subtarget['inds']): # A function has been provided
        subtarget_inds = subtarget['inds'](sim) # Call the function to get the indices
    else:
        subtarget_inds = subtarget['inds'] # The indices are supplied directly

    # Validate the values
    subtarget_vals = subtarget['vals']
    if sc.isiterable(subtarget_vals):
        if len(subtarget_vals) != len(subtarget_inds):
            errormsg = f'Length of subtargeting indices ({len(subtarget_inds)}) does not match length of values ({len(subtarget_vals)})'
            raise ValueError(errormsg)


    return subtarget_inds, subtarget_vals


class test_num(Intervention):
    '''
    Test a fixed number of people per day.

    Args:
        daily_tests (arr)   : number of tests per day, can be int, array, or dataframe/series; if integer, use that number every day
        symp_test   (float) : odds ratio of a symptomatic person testing
        quar_test   (float) : probability of a person in quarantine testing
        subtarget   (dict)  : subtarget intervention to people with particular indices                                                 ( format : {'ind' : array of indices, or function to return indices from the sim, 'vals' : value ( s) to apply}
        sensitivity (float) : test sensitivity
        ili_prev    (arr)   : Prevalence of influenza-like-illness symptoms in the population; can be float, array, or dataframe/series
        loss_prob   (float) : probability of the person being lost-to-follow-up
        test_delay  (int)   : days for test result to be known
        start_day   (int)   : day the intervention starts
        end_day     (int)   : day the intervention ends
        swab_delay  (dict)  : distribution for the delay from onset to swab
        kwargs      (dict)  : passed to Intervention                                                                                   ( )

    **Examples**::

        interv = cv.test_num(daily_tests=[0.10*n_people]*npts)
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': sim.people.age>50, 'vals': 1.2}) # People over 50 are 20% more likely to test
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': lambda sim: sim.people.age>50, 'vals': 1.2}) # People over 50 are 20% more likely to test
    '''

    def __init__(self, daily_tests, symp_test=100.0, quar_test=1.0, subtarget=None,
                 ili_prev=None, sensitivity=1.0, loss_prob=0, test_delay=0,
                 start_day=0, end_day=None, swab_delay=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        self.daily_tests = daily_tests # Should be a list of length matching time
        self.symp_test   = symp_test   # Set probability of testing symptomatics
        self.quar_test   = quar_test
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
            n_tests = int(self.daily_tests[rel_t]/sim.rescale_vec[t])  # Number of tests for this day -- rescaled
            if not (n_tests and pl.isfinite(n_tests)): # If there are no tests today, abort early
                return
            else:
                sim.results['new_tests'][t] += n_tests
        else:
            return

        test_probs = np.ones(sim.n) # Begin by assigning equal testing probability to everyone

        # Calculate test probabilities for people with symptoms
        symp_inds = cvu.true(sim.people.symptomatic)
        symp_test = self.symp_test
        if self.pdf: # Handle the onset to swab delay
            symp_time = int(t - sim.people.date_symptomatic[symp_inds]) # Find time since symptom onset
            inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
            count = np.nan * np.ones(inv_count.shape) #
            count[inv_count != 0] = 1/inv_count[inv_count != 0]
            symp_test *= self.pdf.pdf(symp_time) * count[symp_time] # Put it all together, have to add a small amount because 0.0 will fail

        test_probs[symp_inds] *= symp_test

        # Handle symptomatic testing, taking into account prevalence of ILI symptoms
        if self.ili_prev is not None:
            if rel_t < len(self.ili_prev):
                n_ili = int(self.ili_prev[rel_t] * sim['pop_size'])  # Number with ILI symptoms on this day
                ili_inds = cvu.choose(sim['pop_size'], n_ili) # Give some people some symptoms. Assuming that this is independent of COVID symptomaticity...
                ili_inds = np.setdiff1d(ili_inds, symp_inds)
                test_probs[ili_inds] *= self.symp_test

        # Handle quarantine testing
        quar_inds  = cvu.true(sim.people.quarantined)
        test_probs[quar_inds] *= self.quar_test

        # Handle any other user-specified testing criteria
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            test_probs[subtarget_inds] = test_probs[subtarget_inds]*subtarget_vals

        # Don't re-diagnose people
        diag_inds  = cvu.true(sim.people.diagnosed)
        test_probs[diag_inds] = 0.

        # Now choose who gets tested and test them
        test_inds = cvu.choose_w(probs=test_probs, n=n_tests, unique=False)
        sim.people.test(test_inds, self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)

        return


class test_prob(Intervention):
    '''
    Test as many people as required based on test probability.
    Probabilities are OR together, so choose wisely.

    Args:
        symp_prob (float): Probability of testing a symptomatic (unquarantined) person
        asymp_prob (float): Probability of testing an asymptomatic (unquarantined) person
        symp_quar_prob (float): Probability of testing a symptomatic quarantined person
        asymp_quar_prob (float): Probability of testing an asymptomatic quarantined person
        subtarget (dict): subtarget intervention to people with particular indices (see test_num() for details)
        test_sensitivity (float): Probability of a true positive
        ili_prev (float or array): Prevalence of influenza-like-illness symptoms in the population
        loss_prob (float): Probability of loss to follow-up
        test_delay (int): How long testing takes
        start_day (int): When to start the intervention
        kwargs (dict): passed to Intervention()

    **Examples**::

        interv = cv.test_prob(symp_prob=0.1, asymp_prob=0.01) # Test 10% of symptomatics and 1% of asymptomatics
        interv = cv.test_prob(symp_quar_prob=0.4) # Test 40% of those in quarantine with symptoms
    '''
    def __init__(self, symp_prob, asymp_prob=0.0, symp_quar_prob=None, asymp_quar_prob=None, subtarget=None, ili_prev=None,
                 test_sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0, end_day=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so the intervention can be recreated
        self.symp_prob        = symp_prob
        self.asymp_prob       = asymp_prob
        self.symp_quar_prob   = symp_quar_prob  if  symp_quar_prob is not None else  symp_prob
        self.asymp_quar_prob  = asymp_quar_prob if asymp_quar_prob is not None else asymp_prob
        self.subtarget        = subtarget
        self.ili_prev         = ili_prev
        self.test_sensitivity = test_sensitivity
        self.loss_prob        = loss_prob
        self.test_delay       = test_delay
        self.start_day        = start_day
        self.end_day          = end_day
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

        # Define symptomatics, accounting for ILI prevalence
        symp_inds  = cvu.true(sim.people.symptomatic)
        if self.ili_prev is not None:
            rel_t = t - self.start_day
            if rel_t < len(self.ili_prev):
                n_ili = int(self.ili_prev[rel_t] * sim['pop_size'])  # Number with ILI symptoms on this day
                ili_inds = cvu.choose(sim['pop_size'], n_ili) # Give some people some symptoms, assuming that this is independent of COVID symptomaticity...
                symp_inds = np.unique(np.concatenate((symp_inds, ili_inds)),0)

        # Define asymptomatics: those who neither have COVID symptoms nor ILI symptoms
        asymp_inds = np.setdiff1d(np.arange(sim['pop_size']), symp_inds)

        # Handle quarantine and other testing criteria
        quar_inds       = cvu.true(sim.people.quarantined)
        symp_quar_inds  = np.intersect1d(quar_inds, symp_inds)
        asymp_quar_inds = np.intersect1d(quar_inds, asymp_inds)
        if self.subtarget is not None:
            subtarget_inds  = self.subtarget['inds']
        diag_inds       = cvu.true(sim.people.diagnosed)

        # Construct the testing probabilities piece by piece -- complicated, since need to do it in the right order
        test_probs = np.zeros(sim.n) # Begin by assigning equal testing probability to everyone
        test_probs[symp_inds]       = self.symp_prob       # People with symptoms
        test_probs[asymp_inds]      = self.asymp_prob      # People without symptoms
        test_probs[symp_quar_inds]  = self.symp_quar_prob  # People with symptoms in quarantine
        test_probs[asymp_quar_inds] = self.asymp_quar_prob # People without symptoms in quarantine
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            test_probs[subtarget_inds] = subtarget_vals # People being explicitly subtargeted
        test_probs[diag_inds] = 0.0 # People who are diagnosed don't test
        test_inds = cvu.binomial_arr(test_probs).nonzero()[0] # Finally, calculate who actually tests

        sim.people.test(test_inds, test_sensitivity=self.test_sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)
        sim.results['new_tests'][t] += int(len(test_inds)*sim['pop_scale']/sim.rescale_vec[t]) # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec

        return


class contact_tracing(Intervention):
    '''
    Contact tracing of positive people.

    Args:
        trace_probs (dict): probability of tracing, per layer
        trace_time  (dict): days required to trace, per layer
        start_day   (int):  intervention start day
        end_day     (int):  intervention end day
        test_delay  (int):  number of days a test result takes
        presumptive (bool): whether or not to begin isolation and contact tracing on the presumption of a positive diagnosis
        kwargs      (dict): passed to Intervention()
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
