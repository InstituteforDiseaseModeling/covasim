'''
Defines the Sim class, Covasim's core class.
'''

#%% Imports
import numpy as np
import sciris as sc
from . import version as cvv
from . import utils as cvu
from . import misc as cvm
from . import base as cvb
from . import defaults as cvd
from . import parameters as cvpar
from . import population as cvpop
from . import plotting as cvplt
from . import interventions as cvi
from . import analysis as cva

# Everything in this file is contained in the Sim class
__all__ = ['Sim']


class Sim(cvb.BaseSim):
    '''
    The Sim class handles the running of the simulation: the number of children,
    number of time points, and the parameters of the simulation.

    Args:
        pars     (dict):   parameters to modify from their default values
        datafile (str/df): filename of (Excel, CSV) data file to load, or a pandas dataframe of the data
        datacols (list):   list of column names of the data to load
        label    (str):    the name of the simulation (useful to distinguish in batch runs)
        simfile  (str):    the filename for this simulation, if it's saved (default: creation date)
        popfile  (str):    the filename to load/save the population for this simulation
        load_pop (bool):   whether to load the population from the named file
        save_pop (bool):   whether to save the population to the named file
        kwargs   (dict):   passed to make_pars()

    **Examples**::

        sim = cv.Sim()
        sim = cv.Sim(pop_size=10e3, datafile='my_data.xlsx')
    '''

    def __init__(self, pars=None, datafile=None, datacols=None, label=None, simfile=None, popfile=None, load_pop=False, save_pop=False, **kwargs):
        # Create the object
        default_pars = cvpar.make_pars() # Start with default pars
        super().__init__(default_pars) # Initialize and set the parameters as attributes

        # Set attributes
        self.label         = label    # The label/name of the simulation
        self.created       = None     # The datetime the sim was created
        self.simfile       = simfile  # The filename of the sim
        self.datafile      = datafile # The name of the data file
        self.popfile       = popfile  # The population file
        self.load_pop      = load_pop # Whether to load the population
        self.save_pop      = save_pop # Whether to save the population
        self.data          = None     # The actual data
        self.popdict       = None     # The population dictionary
        self.t             = None     # The current time in the simulation
        self.people        = None     # Initialize these here so methods that check their length can see they're empty
        self.results       = {}       # For storing results
        self.initialized   = False    # Whether or not initialization is complete
        self.results_ready = False    # Whether or not results are ready

        # Now update everything
        self.set_metadata(simfile, label)  # Set the simulation date and filename
        self.update_pars(pars, **kwargs)   # Update the parameters, if provided
        self.load_data(datafile, datacols) # Load the data, if provided
        if self.load_pop:
            self.load_population(popfile)      # Load the population, if provided

        return


    def update_pars(self, pars=None, create=False, **kwargs):
        ''' Ensure that metaparameters get used properly before being updated '''
        pars = sc.mergedicts(pars, kwargs)
        if pars:
            if pars.get('pop_type'):
                cvpar.reset_layer_pars(pars, force=False)
            if pars.get('prog_by_age'):
                pars['prognoses'] = cvpar.get_prognoses(by_age=pars['prog_by_age']) # Reset prognoses
            super().update_pars(pars=pars, create=create) # Call update_pars() for ParsObj
        return


    def set_metadata(self, simfile, label):
        ''' Set the metadata for the simulation -- creation time and filename '''
        self.created = sc.now()
        self.version = cvv.__version__
        self.git_info = cvm.git_info()
        if simfile is None:
            datestr = sc.getdate(obj=self.created, dateformat='%Y-%b-%d_%H.%M.%S')
            self.simfile = f'covasim_{datestr}.sim'
        if label is not None:
            self.label = label
        return


    def load_data(self, datafile=None, datacols=None, verbose=None, **kwargs):
        ''' Load the data to calibrate against, if provided '''
        if verbose is None:
            verbose = self['verbose']
        self.datafile = datafile # Store this
        if datafile is not None: # If a data file is provided, load it
            self.data = cvm.load_data(datafile=datafile, columns=datacols, verbose=verbose, **kwargs)

        return


    def initialize(self, **kwargs):
        '''
        Perform all initializations, including validating the parameters, setting
        the random number seed, creating the results structure, initializing the
        people, validating the layer parameters (which requires the people),
        and initializing the interventions.

        Args:
            kwargs (dict): passed to init_people
        '''
        self.t = 0  # The current time index
        self.validate_pars() # Ensure parameters have valid values
        self.set_seed() # Reset the random seed before the population is created
        self.init_results() # Create the results stucture
        self.init_people(save_pop=self.save_pop, load_pop=self.load_pop, popfile=self.popfile, **kwargs) # Create all the people (slow)
        self.validate_layer_pars() # Once the population is initialized, validate the layer parameters again
        self.init_interventions() # Initialize the interventions
        self.init_analyzers() # ...and the interventions
        self.set_seed() # Reset the random seed again so the random number stream is consistent
        self.initialized = True
        return


    def layer_keys(self):
        '''
        Attempt to retrieve the current layer keys, in the following order: from
        the people object (for an initialized sim), from the popdict (for one in
        the process of being initialized), from the beta_layer parameter (for an
        uninitialized sim), or by assuming a default (if none of the above are
        available).
        '''
        try:
            keys = list(self['beta_layer'].keys()) # Get keys from beta_layer since the "most required" layer parameter
        except:
            keys = []
        return keys


    def reset_layer_pars(self, layer_keys=None, force=False):
        '''
        Reset the parameters to match the population.

        Args:
            layer_keys (list): override the default layer keys (use stored keys by default)
            force (bool): reset the parameters even if they already exist
        '''
        if layer_keys is None:
            if self.people is not None: # If people exist
                layer_keys = self.people.contacts.keys()
            elif self.popdict is not None:
                layer_keys = self.popdict['layer_keys']
        cvpar.reset_layer_pars(self.pars, layer_keys=layer_keys, force=force)
        return


    def validate_layer_pars(self):
        '''
        Handle layer parameters, since they need to be validated after the population
        creation, rather than before.
        '''

        # First, try to figure out what the layer keys should be and perform basic type checking
        layer_keys = self.layer_keys()
        layer_pars = cvpar.layer_pars # The names of the parameters that are specified by layer
        for lp in layer_pars:
            val = self[lp]
            if sc.isnumber(val): # It's a scalar instead of a dict, assume it's all contacts
                self[lp] = {k:val for k in layer_keys}

        # Handle key mismaches
        for lp in layer_pars:
            lp_keys = set(self.pars[lp].keys())
            if not lp_keys == set(layer_keys):
                errormsg = f'Layer parameters have inconsistent keys with the layer keys {layer_keys}:'
                for lp2 in layer_pars: # Fail on first error, but re-loop to list all of them
                    errormsg += f'\n{lp2} = ' + ', '.join(self.pars[lp].keys())
                raise sc.KeyNotFoundError(errormsg)

        # Handle mismatches with the population
        if self.people is not None:
            pop_keys = set(self.people.contacts.keys())
            if pop_keys != set(layer_keys):
                errormsg = f'Please update your parameter keys {layer_keys} to match population keys {pop_keys}. You may find sim.reset_layer_pars() helpful.'
                raise sc.KeyNotFoundError(errormsg)

        return


    def validate_pars(self, validate_layers=True):
        '''
        Some parameters can take multiple types; this makes them consistent.

        Args:
            validate_layers (bool): whether to validate layer parameters as well via validate_layer_pars() -- usually yes, except during initialization
        '''

        # Handle types
        for key in ['pop_size', 'pop_infected', 'pop_size']:
            try:
                self[key] = int(self[key])
            except Exception as E:
                errormsg = f'Could not convert {key}={self[key]} of {type(self[key])} to integer'
                raise ValueError(errormsg) from E

        # Handle start day
        start_day = self['start_day'] # Shorten
        if start_day in [None, 0]: # Use default start day
            start_day = '2020-03-01'
        self['start_day'] = cvm.date(start_day)

        # Handle end day and n_days
        end_day = self['end_day']
        n_days = self['n_days']
        if end_day:
            self['end_day'] = cvm.date(end_day)
            n_days = cvm.daydiff(self['start_day'], self['end_day'])
            if n_days <= 0:
                errormsg = f"Number of days must be >0, but you supplied start={str(self['start_day'])} and end={str(self['end_day'])}, which gives n_days={n_days}"
                raise ValueError(errormsg)
            else:
                self['n_days'] = int(n_days)
        else:
            if n_days:
                self['n_days'] = int(n_days)
                self['end_day'] = self.date(n_days) # Convert from the number of days to the end day
            else:
                errormsg = f'You must supply one of n_days and end_day, not "{n_days}" and "{end_day}"'
                raise ValueError(errormsg)

        # Handle population data
        popdata_choices = ['random', 'hybrid', 'clustered', 'synthpops']
        choice = self['pop_type']
        if choice and choice not in popdata_choices:
            choicestr = ', '.join(popdata_choices)
            errormsg = f'Population type "{choice}" not available; choices are: {choicestr}'
            raise ValueError(errormsg)

        # Handle interventions and analyzers
        self['interventions'] = sc.promotetolist(self['interventions'], keepnone=False)
        for i,interv in enumerate(self['interventions']):
            if isinstance(interv, dict): # It's a dictionary representation of an intervention
                self['interventions'][i] = cvi.InterventionDict(**interv)
        self['analyzers'] = sc.promotetolist(self['analyzers'], keepnone=False)

        # Optionally handle layer parameters
        if validate_layers:
            self.validate_layer_pars()

        return


    def init_results(self):
        '''
        Create the main results structure.
        We differentiate between flows, stocks, and cumulative results
        The prefix "new" is used for flow variables, i.e. counting new events (infections/deaths/recoveries) on each timestep
        The prefix "n" is used for stock variables, i.e. counting the total number in any given state (sus/inf/rec/etc) on any particular timestep
        The prefix "cum" is used for cumulative variables, i.e. counting the total number that have ever been in a given state at some point in the sim
        Note that, by definition, n_dead is the same as cum_deaths and n_recovered is the same as cum_recoveries, so we only define the cumulative versions
        '''

        def init_res(*args, **kwargs):
            ''' Initialize a single result object '''
            output = cvb.Result(*args, **kwargs, npts=self.npts)
            return output

        dcols = cvd.get_colors() # Get default colors

        # Flows and cumulative flows
        for key,label in cvd.result_flows.items():
            self.results[f'cum_{key}'] = init_res(f'Cumulative {label}',    color=dcols[key]) # Cumulative variables -- e.g. "Cumulative infections"

        for key,label in cvd.result_flows.items(): # Repeat to keep all the cumulative keys together
            self.results[f'new_{key}'] = init_res(f'Number of new {label}', color=dcols[key]) # Flow variables -- e.g. "Number of new infections"

        # Stock variables
        for key,label in cvd.result_stocks.items():
            self.results[f'n_{key}'] = init_res(label, color=dcols[key])
        self.results['n_susceptible'].scale = 'static'

        # Other variables
        self.results['prevalence']    = init_res('Prevalence', scale=False)
        self.results['incidence']     = init_res('Incidence', scale=False)
        self.results['r_eff']         = init_res('Effective reproductive number', scale=False)
        self.results['doubling_time'] = init_res('Doubling time', scale=False)
        self.results['test_yield']    = init_res('Testing yield', scale=False)

        # Populate the rest of the results
        if self['rescale']:
            scale = 1
        else:
            scale = self['pop_scale']
        self.rescale_vec   = scale*np.ones(self.npts) # Not included in the results, but used to scale them
        self.results['t']    = self.tvec
        self.results['date'] = self.datevec
        self.results_ready   = False

        return


    def load_population(self, popfile=None, **kwargs):
        '''
        Load the population dictionary from file -- typically done automatically
        as part of sim.initialize(). Supports loading either saved population
        dictionaries (popdicts, file ending .pop by convention), or ready-to-go
        People objects (file ending .ppl by convention). Either object an also be
        supplied directly. Once a population file is loaded, it is removed from
        the Sim object.

        Args:
            popfile (str or obj): if a string, name of the file; otherwise, the popdict or People object to load
            kwargs (dict): passed to sc.makefilepath()
        '''
        # Set the file path if not is provided
        if popfile is None and self.popfile is not None:
            popfile = self.popfile

        # Handle the population (if it exists)
        if popfile is not None:

            # Load from disk or use directly
            if isinstance(popfile, str): # It's a string, assume it's a filename
                filepath = sc.makefilepath(filename=popfile, **kwargs)
                obj = sc.loadobj(filepath)
                if self['verbose']:
                    print(f'Loading population from {filepath}')
            else:
                obj = popfile # Use it directly

            # Process the input
            if isinstance(obj, dict):
                self.popdict = obj
                n_actual     = len(self.popdict['uid'])
                layer_keys   = self.popdict['layer_keys']
            elif isinstance(obj, cvb.BasePeople):
                self.people = obj
                self.people.pars = self.pars # Replace the saved parameters with this simulation's
                n_actual    = len(self.people)
                layer_keys  = self.people.layer_keys()
            else:
                errormsg = f'Cound not interpret input of {type(obj)} as a population file: must be a dict or People object'
                raise ValueError(errormsg)

            # Perform validation
            n_expected = self['pop_size']
            if n_actual != n_expected:
                errormsg = f'Wrong number of people ({n_expected:n} requested, {n_actual:n} actual) -- please change "pop_size" to match or regenerate the file'
                raise ValueError(errormsg)
            self.reset_layer_pars(force=False, layer_keys=layer_keys) # Ensure that layer keys match the loaded population
            self.popfile = None # Once loaded, remove to save memory

        return


    def init_people(self, save_pop=False, load_pop=False, popfile=None, verbose=None, **kwargs):
        '''
        Create the people.

        Args:
            save_pop (bool): if true, save the population dictionary to popfile
            load_pop (bool): if true, load the population dictionary from popfile
            popfile (str): filename to load/save the population
            verbose (int): detail to print
            kwargs (dict): passed to cv.make_people()
        '''

        # Handle inputs
        if verbose is None:
            verbose = self['verbose']
        if verbose:
            print(f'Initializing sim with {self["pop_size"]:0n} people for {self["n_days"]} days')
        if load_pop and self.popdict is None:
            self.load_population(popfile=popfile)

        # Actually make the people
        self.people = cvpop.make_people(self, save_pop=save_pop, popfile=popfile, verbose=verbose, **kwargs)
        self.people.initialize() # Fully initialize the people

        # Create the seed infections
        inds = cvu.choose(self['pop_size'], self['pop_infected'])
        self.people.infect(inds=inds, layer='seed_infection')
        return


    def init_interventions(self):
        ''' Initialize the interventions '''
        for intervention in self['interventions']:
            if isinstance(intervention, cvi.Intervention):
                if not intervention.initialized:
                    intervention.initialize(self)
        return


    def init_analyzers(self):
        ''' Initialize the analyzers '''
        for analyzer in self['analyzers']:
            if isinstance(analyzer, cva.Analyzer):
                if not analyzer.initialized:
                    analyzer.initialize(self)
        return


    def rescale(self):
        ''' Dynamically rescale the population -- used during step() '''
        if self['rescale']:
            t = self.t
            pop_scale = self['pop_scale']
            current_scale = self.rescale_vec[t]
            if current_scale < pop_scale: # We have room to rescale
                n_not_sus = self.people.count_not('susceptible')
                n_people = len(self.people)
                if n_not_sus / n_people > self['rescale_threshold']: # Check if we've reached point when we want to rescale
                    max_ratio = pop_scale/current_scale # We don't want to exceed this
                    scaling_ratio = min(self['rescale_factor'], max_ratio)
                    self.rescale_vec[t:] *= scaling_ratio # Update the rescaling factor from here on
                    n = int(n_people*(1.0-1.0/scaling_ratio)) # For example, rescaling by 2 gives n = 0.5*n_people
                    new_sus_inds = cvu.choose(max_n=n_people, n=n) # Choose who to make susceptible again
                    self.people.make_susceptible(new_sus_inds)
        return


    def step(self):
        '''
        Step the simulation forward in time. Usually, the user would use sim.run()
        rather than calling sim.step() directly.
        '''

        # Set the time and if we have reached the end of the simulation, then do nothing
        t = self.t
        if t >= self.npts:
            return

        # Perform initial operations
        self.rescale() # Check if we need to rescale
        people   = self.people # Shorten this for later use
        people.update_states_pre(t=t) # Update the state of everyone and count the flows
        contacts = people.update_contacts() # Compute new contacts
        hosp_max = people.count('severe')   > self['n_beds_hosp'] if self['n_beds_hosp'] else False # Check for acute bed constraint
        icu_max  = people.count('critical') > self['n_beds_icu']  if self['n_beds_icu']  else False # Check for ICU bed constraint

        # Randomly infect some people (imported infections)
        n_imports = cvu.poisson(self['n_imports']) # Imported cases
        if n_imports>0:
            importation_inds = cvu.choose(max_n=len(people), n=n_imports)
            people.infect(inds=importation_inds, hosp_max=hosp_max, icu_max=icu_max, layer='importation')

        # Apply interventions
        for intervention in self['interventions']:
            if isinstance(intervention, cvi.Intervention):
                intervention.apply(self) # If it's an intervention, call the apply() method
            elif callable(intervention):
                intervention(self) # If it's a function, call it directly
            else:
                errormsg = f'Intervention {intervention} is neither callable nor an Intervention object'
                raise ValueError(errormsg)

        people.update_states_post() # Check for state changes after interventions

        # Compute the probability of transmission
        beta         = cvd.default_float(self['beta'])
        asymp_factor = cvd.default_float(self['asymp_factor'])
        frac_time    = cvd.default_float(self['viral_dist']['frac_time'])
        load_ratio   = cvd.default_float(self['viral_dist']['load_ratio'])
        high_cap     = cvd.default_float(self['viral_dist']['high_cap'])
        date_inf     = people.date_infectious
        date_rec     = people.date_recovered
        date_dead    = people.date_dead
        viral_load = cvu.compute_viral_load(t, date_inf, date_rec, date_dead, frac_time, load_ratio, high_cap)

        for lkey,layer in contacts.items():
            p1 = layer['p1']
            p2 = layer['p2']
            betas   = layer['beta']

            # Compute relative transmission and susceptibility
            rel_trans   = people.rel_trans
            rel_sus     = people.rel_sus
            inf         = people.infectious
            sus         = people.susceptible
            symp        = people.symptomatic
            diag        = people.diagnosed
            quar        = people.quarantined
            iso_factor  = cvd.default_float(self['iso_factor'][lkey])
            quar_factor = cvd.default_float(self['quar_factor'][lkey])
            beta_layer  = cvd.default_float(self['beta_layer'][lkey])
            rel_trans, rel_sus = cvu.compute_trans_sus(rel_trans, rel_sus, inf, sus, beta_layer, viral_load, symp, diag, quar, asymp_factor, iso_factor, quar_factor)

            # Calculate actual transmission
            for sources,targets in [[p1,p2], [p2,p1]]: # Loop over the contact network from p1->p2 and p2->p1
                source_inds, target_inds = cvu.compute_infections(beta, sources, targets, betas, rel_trans, rel_sus) # Calculate transmission!
                people.infect(inds=target_inds, hosp_max=hosp_max, icu_max=icu_max, source=source_inds, layer=lkey) # Actually infect people

        # Update counts for this time step: stocks
        for key in cvd.result_stocks.keys():
            self.results[f'n_{key}'][t] = people.count(key)

        # Update counts for this time step: flows
        for key,count in people.flows.items():
            self.results[key][t] += count

        # Apply analyzers -- same syntax as interventions
        for analyzer in self['analyzers']:
            if isinstance(analyzer, cva.Analyzer):
                analyzer.apply(self) # If it's an intervention, call the apply() method
            elif callable(analyzer):
                analyzer(self) # If it's a function, call it directly
            else:
                errormsg = f'Analyzer {analyzer} is neither callable nor an Analyzer object'
                raise ValueError(errormsg)

        # Tidy up
        self.t += 1
        return


    def run(self, do_plot=False, until=None, verbose=None, restore_pars=True, reset_seed=True, **kwargs):
        '''
        Run the simulation.

        Args:
            do_plot (bool): whether to plot
            until (int): day to run until
            verbose (int): level of detail to print (otherwise uses self['verbose'])
            restore_pars (bool): whether to make a copy of the parameters before the run and restore it after, so runs are repeatable
            reset_seed (bool): whether to reset the random number stream immediately before run
            kwargs (dict): passed to self.plot()

        Returns:
            results: the results object (also modifies in-place)
        '''

        # Initialize
        T = sc.tic()
        if not self.initialized:
            self.initialize()
        else:
            self.validate_pars() # We always want to validate the parameters before running
            self.init_interventions() # And interventions
            if reset_seed:
                self.set_seed() # Ensure the random number generator is freshly initialized
        if restore_pars:
            orig_pars = sc.dcp(self.pars) # Create a copy of the parameters, to restore after the run, in case they are dynamically modified
        if verbose is None:
            verbose = self['verbose']
        if until:
            until = self.day(until)

        # Main simulation loop
        for t in self.tvec:

            # Print progress
            if verbose >= 1:
                elapsed = sc.toc(output=True)
                simlabel = f'"{self.label}": ' if self.label else ''
                string = f'  Running {simlabel}{self.datevec[t]} ({t:2.0f}/{self.pars["n_days"]}) ({elapsed:0.2f} s) '
                if verbose >= 2:
                    sc.heading(string)
                elif verbose == 1:
                    sc.progressbar(t+1, self.npts, label=string, length=20, newline=True)

            # Do the heavy lifting -- actually run the model!
            self.step()

            # Check if we were asked to stop
            elapsed = sc.toc(T, output=True)
            if self['timelimit'] and elapsed > self['timelimit']:
                sc.printv(f"Time limit ({self['timelimit']} s) exceeded", 1, verbose)
                break
            elif self['stopping_func'] and self['stopping_func'](self):
                sc.printv("Stopping function terminated the simulation", 1, verbose)
                break
            if self.t == until: # If until is specified, just stop here
                return

        # End of time loop; compute cumulative results outside of the time loop
        self.finalize(verbose=verbose) # Finalize the results
        sc.printv(f'Run finished after {elapsed:0.2f} s.\n', 1, verbose)
        if restore_pars:
            self.restore_pars(orig_pars)
        if do_plot: # Optionally plot
            self.plot(**kwargs)

        return self.results


    def restore_pars(self, orig_pars):
        ''' Restore the original parameter values, except for the analyzers '''
        analyzers = self['analyzers'] # Make a copy so these don't get wiped
        self.pars = orig_pars # Restore the original parameters
        self['analyzers'] = analyzers # Restore the analyzers
        return


    def finalize(self, verbose=None):
        ''' Compute final results '''

        # Scale the results
        for reskey in self.result_keys():
            if self.results[reskey].scale == 'dynamic':
                self.results[reskey].values *= self.rescale_vec
            elif self.results[reskey].scale == 'static':
                self.results[reskey].values *= self['pop_scale']

        # Calculate cumulative results
        for key in cvd.result_flows.keys():
            self.results[f'cum_{key}'].values[:] = np.cumsum(self.results[f'new_{key}'].values)
        self.results['cum_infections'].values += self['pop_infected']*self.rescale_vec[0] # Include initially infected people

        # Final settings
        self.t -= 1 # During the run, this keeps track of the next step; restore this be the final day of the sim
        self.results_ready = True # Set this first so self.summary() knows to print the results
        self.initialized = False # To enable re-running

        # Perform calculations on results
        self.compute_results(verbose=verbose) # Calculate the rest of the results
        self.results = sc.objdict(self.results) # Convert results to a odicts/objdict to allow e.g. sim.results.diagnoses

        return


    def compute_results(self, verbose=None):
        ''' Perform final calculations on the results '''
        self.compute_prev_inci()
        self.compute_yield()
        self.compute_doubling()
        self.compute_r_eff()
        self.compute_summary(verbose=verbose)
        return


    def compute_prev_inci(self):
        '''
        Compute prevalence and incidence. Prevalence is the current number of infected
        people divided by the number of people who are alive. Incidence is the number
        of new infections per day divided by the susceptible population.
        '''
        n_exposed = self.results['n_exposed'].values # Number of people currently infected
        n_alive = self.scaled_pop_size - self.results['cum_deaths'].values # Number of people still alive
        n_susceptible = self.results['n_susceptible'].values # Number of people still susceptible
        new_infections = self.results['new_infections'].values # Number of new infections
        self.results['prevalence'][:] = n_exposed/n_alive # Calculate the prevalence
        self.results['incidence'][:] = new_infections/n_susceptible # Calculate the incidence
        return


    def compute_yield(self):
        ''' Compute test yield -- number of positive tests divided by the total number of tests '''
        n_diags = self.results['new_diagnoses'].values # Number of positive tests
        n_tests = self.results['new_tests'].values # Total number of tests
        inds = cvu.true(n_tests) # Pull out non-zero numbers of tests
        self.results['test_yield'].values[inds] = n_diags[inds]/n_tests[inds] # Calculate the yield
        return


    def compute_doubling(self, window=3, max_doubling_time=30):
        '''
        Calculate doubling time using exponential approximation -- a more detailed
        approach is in utils.py. Compares infections at time t to infections at time
        t-window, and uses that to compute the doubling time. For example, if there are
        100 cumulative infections on day 12 and 200 infections on day 19, doubling
        time is 7 days.

        Args:
            window (float): the size of the window used (larger values are more accurate but less precise)
            max_doubling_time (float): doubling time could be infinite, so this places a bound on it

        Returns:
            doubling_time (array): the doubling time results array
        '''
        cum_infections = self.results['cum_infections'].values
        self.results['doubling_time'][:window] = np.nan
        for t in range(window, self.npts):
            infections_now = cum_infections[t]
            infections_prev = cum_infections[t-window]
            r = infections_now/infections_prev
            if r > 1:  # Avoid divide by zero
                doubling_time = window*np.log(2)/np.log(r)
                doubling_time = min(doubling_time, max_doubling_time) # Otherwise, it's unbounded
                self.results['doubling_time'][t] = doubling_time
        return self.results['doubling_time'].values


    def compute_r_eff(self, method='daily', smoothing=2, window=7):
        '''
        Effective reproductive number based on number of people each person infected.

        Args:
            method (str): 'instant' uses daily infections, 'infectious' counts from the date infectious, 'outcome' counts from the date recovered/dead
            smoothing (int): the number of steps to smooth over for the 'daily' method
            window (int): the size of the window used for 'infectious' and 'outcome' calculations (larger values are more accurate but less precise)

        Returns:
            r_eff (array): the r_eff results array
        '''

        # Initialize arrays to hold sources and targets infected each day
        sources = np.zeros(self.npts)
        targets = np.zeros(self.npts)
        window = int(window)

        # Default method -- calculate the daily infections
        if method == 'daily':

            # Find the dates that everyone became infectious and recovered, and hence calculate infectious duration
            recov_inds   = self.people.defined('date_recovered')
            dead_inds    = self.people.defined('date_dead')
            date_recov   = self.people.date_recovered[recov_inds]
            date_dead    = self.people.date_dead[dead_inds]
            date_outcome = np.concatenate((date_recov, date_dead))
            inds         = np.concatenate((recov_inds, dead_inds))
            date_inf     = self.people.date_infectious[inds]
            mean_inf     = date_outcome.mean() - date_inf.mean()

            # Calculate R_eff as the mean infectious duration times the number of new infectious divided by the number of infectious people on a given day
            raw_values = mean_inf*self.results['new_infections'].values/(self.results['n_infectious'].values+1e-6)
            len_raw = len(raw_values) # Calculate the number of raw values
            if len_raw >= 3: # Can't smooth arrays shorter than this since the default smoothing kernel has length 3
                initial_period = self['dur']['exp2inf']['par1'] + self['dur']['asym2rec']['par1'] # Approximate the duration of the seed infections for averaging
                initial_period = int(min(len_raw, initial_period)) # Ensure we don't have too many points
                for ind in range(initial_period): # Loop over each of the initial inds
                    raw_values[ind] = raw_values[ind:initial_period].mean() # Replace these values with their average
                values = sc.smooth(raw_values, smoothing)
                values[:smoothing] = raw_values[:smoothing] # To avoid numerical effects, replace the beginning and end with the original
                values[-smoothing:] = raw_values[-smoothing:]
            else:
                values = raw_values

        # Alternate (traditional) method -- count from the date of infection or outcome
        elif method in ['infectious', 'outcome']:

            # Store a mapping from each source to their date
            source_dates = {}

            for t in self.tvec:

                # Sources are easy -- count up the arrays for all the people who became infections on that day
                if method == 'infectious':
                    inds = cvu.true(t == self.people.date_infectious) # Find people who became infectious on this timestep
                elif method == 'outcome':
                    recov_inds = cvu.true(t == self.people.date_recovered) # Find people who recovered on this timestep
                    dead_inds  = cvu.true(t == self.people.date_dead)  # Find people who died on this timestep
                    inds       = np.concatenate((recov_inds, dead_inds))
                sources[t] = len(inds)

                # Create the mapping from sources to dates
                for ind in inds:
                    source_dates[ind] = t

            # Targets are hard -- loop over the transmission tree
            for transdict in self.people.infection_log:
                source = transdict['source']
                if source is not None and source in source_dates: # Skip seed infections and people with e.g. recovery after the end of the sim
                    source_date = source_dates[source]
                    targets[source_date] += 1

                # for ind in inds:
                #     targets[t] += len(self.people.transtree.targets[ind])

            # Populate the array -- to avoid divide-by-zero, skip indices that are 0
            r_eff = np.divide(targets, sources, out=np.full(self.npts, np.nan), where=sources > 0)

            # Use stored weights calculate the moving average over the window of timesteps, n
            num = np.nancumsum(r_eff * sources)
            num[window:] = num[window:] - num[:-window]
            den = np.cumsum(sources)
            den[window:] = den[window:] - den[:-window]
            values = np.divide(num, den, out=np.full(self.npts, np.nan), where=den > 0)

        # Method not recognized
        else:
            errormsg = f'Method must be "daily", "infected", or "outcome", not "{method}"'
            raise ValueError(errormsg)

        # Set the values and return
        self.results['r_eff'].values[:] = values

        return self.results['r_eff'].values


    def compute_gen_time(self):
        '''
        Calculate the generation time (or serial interval). There are two
        ways to do this calculation. The 'true' interval (exposure time to
        exposure time) or 'clinical' (symptom onset to symptom onset).

        Returns:
            gen_time (dict): the generation time results
        '''

        intervals1 = np.zeros(len(self.people))
        intervals2 = np.zeros(len(self.people))
        pos1 = 0
        pos2 = 0
        date_exposed = self.people.date_exposed
        date_symptomatic = self.people.date_symptomatic

        for infection in self.people.infection_log:
            if infection['source'] is not None:
                source_ind = infection['source']
                target_ind = infection['target']
                intervals1[pos1] = date_exposed[target_ind] - date_exposed[source_ind]
                pos1 += 1
                if np.isfinite(date_symptomatic[source_ind]) and np.isfinite(date_symptomatic[target_ind]):
                    intervals2[pos2] = date_symptomatic[target_ind] - date_symptomatic[source_ind]
                    pos2 += 1

        self.results['gen_time'] = {
                'true':         np.mean(intervals1[:pos1]),
                'true_std':     np.std(intervals1[:pos1]),
                'clinical':     np.mean(intervals2[:pos2]),
                'clinical_std': np.std(intervals2[:pos2])}
        return self.results['gen_time']


    def compute_summary(self, verbose=None):
        ''' Compute the summary statistics to display at the end of a run '''

        if verbose is None:
            verbose = self['verbose']

        self.summary = sc.objdict()
        for key in self.result_keys():
            self.summary[key] = self.results[key][-1]

        if verbose:
            self.summarize()

        return self.summary


    def summarize(self, output=False):
        ''' Print a brief summary of the simulation '''
        if self.results_ready:
            summary_str = 'Simulation summary:\n'
            for key in self.result_keys():
                if key.startswith('cum_'):
                    summary_str += f'   {self.summary[key]:5.0f} {self.results[key].name.lower()}\n'

            if not output:
                print(summary_str)
            else:
                return summary_str
        else:
            return self.brief(output=output) # If the simulation hasn't been run, default to the brief summary



    def brief(self, output=False):
        ''' Return a one-line description of a sim '''

        if self.results_ready:
            infections = self.summary['cum_infections']
            deaths = self.summary['cum_deaths']
            results = f'{infections:n}⚙, {deaths:n}☠'
        else:
            results = 'not run'

        if self.label:
            label = f'"{self.label}"'
        else:
            label = '<no label>'

        start = cvm.date(self['start_day'], as_date=False)
        if self['end_day']:
            end = cvm.date(self['end_day'], as_date=False)
        else:
            end = cvm.date(self['n_days'], start_date=start)

        pop_size = self['pop_size']
        pop_type = self['pop_type']
        string   = f'Sim({label}; {start}—{end}; pop: {pop_size:n} {pop_type}; epi: {results})'

        if not output:
            print(string)
        else:
            return string


    def compute_fit(self, output=True, *args, **kwargs):
        '''
        Compute the fit between the model and the data. See cv.Fit() for more
        information.

        Args:
            output (bool): whether or not to return the TransTree; if not, store in sim.results
            args   (list): passed to cv.Fit()
            kwargs (dict): passed to cv.Fit()

        **Example**::

            sim = cv.Sim(datafile=data.csv)
            sim.run()
            fit = sim.compute_fit()
            fit.plot()
        '''
        fit = cva.Fit(self, *args, **kwargs)
        if output:
            return fit
        else:
            self.results.fit = fit
            return


    def make_age_histogram(self, output=True, *args, **kwargs):
        '''
        Calculate the age histograms of infections, deaths, diagnoses, etc. See
        cv.age_histogram() for more information. This can be used alternatively
        to supplying the age histogram as an analyzer to the sim. If used this
        way, it can only record the final time point since the states of each
        person are not saved during the sim.

        Args:
            output (bool): whether or not to return the age histogram; if not, store in sim.results
            args   (list): passed to cv.age_histogram()
            kwargs (dict): passed to cv.age_histogram()

        **Example**::

            sim = cv.Sim()
            sim.run()
            agehist = sim.make_age_histogram()
            fiagehistt.plot()
        '''
        agehist = cva.make_age_histogram(self, *args, **kwargs)
        if output:
            return agehist
        else:
            self.results.agehist = agehist
            return


    def make_transtree(self, output=True, *args, **kwargs):
        '''
        Create a TransTree (transmission tree) object, for analyzing the pattern
        of transmissions in the simulation. See cv.TransTree() for more information.

        Args:
            output (bool): whether or not to return the TransTree; if not, store in sim.results
            args   (list): passed to cv.TransTree()
            kwargs (dict): passed to cv.TransTree()

        **Example**::

            sim = cv.Sim()
            sim.run()
            tt = sim.make_transtree()
        '''
        tt = cva.TransTree(self, *args, **kwargs)
        if output:
            return tt
        else:
            self.results.transtree = tt
            return


    def plot(self, *args, **kwargs):
        '''
        Plot the results of a single simulation.

        Args:
            to_plot      (dict): Dict of results to plot; see get_sim_plots() for structure
            do_save      (bool): Whether or not to save the figure
            fig_path     (str):  Path to save the figure
            fig_args     (dict): Dictionary of kwargs to be passed to pl.figure()
            plot_args    (dict): Dictionary of kwargs to be passed to pl.plot()
            scatter_args (dict): Dictionary of kwargs to be passed to pl.scatter()
            axis_args    (dict): Dictionary of kwargs to be passed to pl.subplots_adjust()
            legend_args  (dict): Dictionary of kwargs to be passed to pl.legend(); if show_legend=False, do not show
            show_args    (dict): Control which "extras" get shown: uncertainty bounds, data, interventions, ticks, and the legend
            as_dates     (bool): Whether to plot the x-axis as dates or time points
            dateformat   (str):  Date string format, e.g. '%B %d'
            interval     (int):  Interval between tick marks
            n_cols       (int):  Number of columns of subpanels to use for subplot
            font_size    (int):  Size of the font
            font_family  (str):  Font face
            grid         (bool): Whether or not to plot gridlines
            commaticks   (bool): Plot y-axis with commas rather than scientific notation
            setylim      (bool): Reset the y limit to start at 0
            log_scale    (bool): Whether or not to plot the y-axis with a log scale; if a list, panels to show as log
            do_show      (bool): Whether or not to show the figure
            colors       (dict): Custom color for each result, must be a dictionary with one entry per result key in to_plot
            sep_figs     (bool): Whether to show separate figures for different results instead of subplots
            fig          (fig):  Handle of existing figure to plot into

        Returns:
            fig: Figure handle

        **Example**::

            sim = cv.Sim()
            sim.run()
            sim.plot()
        '''
        fig = cvplt.plot_sim(sim=self, *args, **kwargs)
        return fig


    def plot_result(self, key, *args, **kwargs):
        '''
        Simple method to plot a single result. Useful for results that aren't
        standard outputs. See sim.plot() for explanation of other arguments.

        Args:
            key (str): the key of the result to plot

        **Examples**::

            sim.plot_result('r_eff')
        '''
        fig = cvplt.plot_result(sim=self, key=key, *args, **kwargs)
        return fig
