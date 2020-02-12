'''
This file contains all the code for a single run of Covid-ABM.

Based heavily on LEMOD-FP (https://github.com/amath-idm/lemod_fp).
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import pylab as pl
import sciris as sc
import numba as nb
from scipy import interpolate as si


# Specify all externally visible things this file defines
__all__ = ['bt', 'bc', 'rbt', 'mt', 'set_seed', 'fixaxis', 'ParsObj', 'Person', 'Sim', 'multi_run']

# Parameters -- should not need to be modified
usenumba   = True  # Whether to use Numba or Numpy functions for random number generators
eps        = 1e-12 # To avoid divide-by-zero errors
   
# Decide which numerical function to use
if usenumba:
    func_decorator = nb.njit
    class_decorator = nb.jitclass # Not used currently
else:
    def func_decorator(*args, **kwargs):
        def wrap(func): return func
        return wrap
    def class_decorator(*args, **kwargs):
        def wrap(cls): return cls
        return wrap



#%% Define helper functions

def set_seed(seed=None):
    ''' Reset the random seed -- complicated because of Numba '''
    
    @func_decorator
    def set_seed_numba(seed):
        return np.random.seed(seed)
    
    def set_seed_regular(seed):
        return np.random.seed(seed)
    
    if seed is not None:
        set_seed_numba(seed)
        set_seed_regular(seed)
    return


@func_decorator((nb.float64,)) # These types can also be declared as a dict, but performance is much slower...?
def bt(prob):
    ''' A simple Bernoulli (binomial) trial '''
    return np.random.random() < prob # Or rnd.random() < prob, np.random.binomial(1, prob), which seems slower

@func_decorator((nb.float64, nb.int64))
def bc(prob, repeats):
    ''' A binomial count '''
    return np.random.binomial(repeats, prob) # Or (np.random.rand(repeats) < prob).sum()

@func_decorator((nb.float64, nb.int64))
def rbt(prob, repeats):
    ''' A repeated Bernoulli (binomial) trial '''
    return np.random.binomial(repeats, prob)>0 # Or (np.random.rand(repeats) < prob).any()

@func_decorator((nb.float64[:],))
def mt(probs):
    ''' A multinomial trial '''
    return np.searchsorted(np.cumsum(probs), np.random.random())


def fixaxis(useSI=True):
    ''' Fix the plotting '''
    pl.legend() # Add legend
    sc.setylim() # Rescale y to start at 0
    return



#%% Define classes
class ParsObj(sc.prettyobj):
    '''
    A class based around performing operations on a self.pars dict.
    '''

    def __init__(self, pars):
        self.update_pars(pars)
        return

    def __getitem__(self, key):
        ''' Allow sim['par_name'] instead of sim.pars['par_name'] '''
        return self.pars[key]

    def __setitem__(self, key, value):
        ''' Ditto '''
        self.pars[key] = value
        self.update_pars()
        return

    def update_pars(self, pars=None):
        ''' Update internal dict with new pars '''
        if not hasattr(self, 'pars'):
            if pars is None:
                raise Exception('Must call update_pars either with a pars dict or with existing pars')
            else:
                self.pars = pars
        elif pars is not None:
            self.pars.update(pars)
        # self._pars_to_attrs() # Convert parameters to attributes
        self._calculate_derived() # Calculate derived parameters
        return
    
    def _calculate_derived(self):
        ''' Calculate derived parameters -- used by update_pars '''
        self.npts = int((self.pars['end'] - self.pars['start'])/self.pars['timestep'] + 1)
        self.tvec = self.pars['start'] + np.arange(self.npts)*self.pars['timestep']
        return




class Person(ParsObj):
    '''
    Class for a single person.
    '''
    def __init__(self, pars, age=0, sex=0, crew=False):
        self.update_pars(pars) # Set parameters
        self.uid = str(pl.randint(0,1e9)) # Unique identifier for this person
        self.age = float(age) # Age of the person (in years)
        self.sex = sex # Female (0) or male (1)
        self.crew       = crew # Wehther the person is a crew member
        self.on_ship    = True # Whether the person is still on the ship
        self.alive      = True
        self.exposed    = False
        self.infectious = False
        self.diagnosed  = False
        self.recovered  = False
        return

    def update(self, t):
        ''' Update the person's state for the given timestep '''
        
        # Initialize outputs
        state = None
        
        dt = self.pars['timestep']
        # TODO -- write person loop

            
        return state



class Sim(ParsObj):
    '''
    The Sim class handles the running of the simulation: the number of children,
    number of time points, and the parameters of the simulation.
    '''

    def __init__(self, pars=None):
        super().__init__(pars) # Initialize and set the parameters as attributes
        set_seed(self.pars['seed'])
        self.init_results()
        self.init_people()
        self.interventions = {}
        return


    def init_results(self):
        resultscols = ['t', 'n_susceptible', 'infections', 'diagnoses', 'deaths']
        self.results = {}
        for key in resultscols:
            self.results[key] = np.zeros(int(self.npts))
        return
    

    def init_people(self):
        ''' Create the people '''
        # TODO age, sex
        self.people = sc.odict() # Dictionary for storing the people
        for i in range(int(self.pars['n'])): # Loop over each person
            person = Person(self.pars, age=age, sex=sex) # Create the person
            self.people[person.uid] = person # Save them to the dictionary
        return

    
    def day2ind(self, day):
        index = int(day/self.pars['timestep'])
        return index
    
    
    def ind2day(self, ind):
        day = ind*self.pars['timestep']
        return day
    
    
    def add_intervention(self, intervention, day):
        index = self.day2ind(day)
        self.interventions[index] = intervention
        return


    def run(self, verbose=None):
        ''' Run the simulation '''
        
        T = sc.tic()
        
        # Reset settings and results
        if verbose is not None:
            self.pars['verbose'] = verbose
        self.update_pars()
        self.init_results()
        self.init_people() # Actually create the children
        
        # Main simulation loop
        for i in range(self.npts):
            t = self.ind2day(i)
            if self.pars['verbose']>-1:
                if sc.approx(t, int(t), eps=0.01):
                    print(f'  Running {t:0.0f} of {self.pars["end"]}...')
            
            # Update each person
            counts = {}
            for person in self.people.values():
                person.update(t, counts) # Update and count new cases
            
            if i in self.interventions:
                self.interventions[i](self)
            
            # Store results
            self.results['t'][i] = self.tvec[i]
            self.results['n'][i]   = self.n
            # TODO
            
        elapsed = sc.toc(T, output=True)
        print(f'Run finished for "{self.pars["name"]}" after {elapsed:0.1f} s')
        return self.results

    
    def plot(self, dosave=None, figargs=None, plotargs=None, axisargs=None, as_days=True):
        '''
        Plot the results -- can supply arguments for both the figure and the plots.

        Parameters
        ----------
        dosave : bool or str
            Whether or not to save the figure. If a string, save to that filename.

        figargs : dict
            Dictionary of kwargs to be passed to pl.figure()

        plotargs : dict
            Dictionary of kwargs to be passed to pl.plot()
        
        as_days : bool
            Whether to plot the x-axis as days or time points

        Returns
        -------
        Figure handle

        '''

        if figargs  is None: figargs  = {'figsize':(26,16)}
        if plotargs is None: plotargs = {'lw':2, 'alpha':0.7, 'marker':'o'}
        if axisargs is None: axisargs = {'left':0.1, 'bottom':0.05, 'right':0.9, 'top':0.97, 'wspace':0.2, 'hspace':0.25}

        fig = pl.figure(**figargs)
        pl.subplots_adjust(**axisargs)

        res = self.results # Shorten since heavily used

        x = res['t'] # Likewise
        if not as_days:
            x /= self.pars['timestep']
            x -= x[0]
            timelabel = 'Timestep'
        else:
            timelabel = 'Day'

        # Plot everything
        to_plot = sc.odict({ # TODO
            'Population size': sc.odict({'pop_size':'Population size'}),
            'MCPR': sc.odict({'mcpr':'Modern contraceptive prevalence rate (%)'}),
            'Births and deaths': sc.odict({'births':'Births', 'deaths':'Deaths'}),
            'Birth-related mortality': sc.odict({'maternal_deaths':'Cumulative birth-related maternal deaths', 'child_deaths':'Cumulative neonatal deaths'}),
            })
        for p,title,keylabels in to_plot.enumitems():
            pl.subplot(2,2,p+1)
            for i,key,label in keylabels.enumitems():
                if label.startswith('Cumulative'):
                    y = pl.cumsum(res[key])
                elif key == 'mcpr':
                    y = res[key]*100
                else:
                    y = res[key]
                pl.plot(x, y, label=label, **plotargs)
            fixaxis()
            if key == 'mcpr':
                pl.ylabel('Percentage')
            else:
                pl.ylabel('Count')
            pl.xlabel(timelabel)
            pl.title(title, fontweight='bold')

        # Ensure the figure actually renders or saves
        if dosave:
            if isinstance(dosave, str):
                filename = dosave # It's a string, assume it's a filename
            else:
                filename = 'voi_sim.png' # Just give it a default name
            pl.savefig(filename)
        else:
            pl.show() # Only show if we're not saving

        return fig
    
    
    def plot_people(self):
        ''' Use imshow() to show all individuals as rows, with time as columns, one pixel per timestep per person '''
        raise NotImplementedError


def single_run(sim):
    sim.run()
    return sim


def multi_run(orig_sim, n=4, verbose=None):
    
    # Copy the simulations
    sims = []
    for i in range(n):
        new_sim = sc.dcp(orig_sim)
        new_sim.pars['seed'] += i # Reset the seed, otherwise no point!
        new_sim.pars['n'] = int(new_sim.pars['n']/n) # Reduce the population size accordingly
        sims.append(new_sim)
        
    finished_sims = sc.parallelize(single_run, iterarg=sims)
    
    output_sim = sc.dcp(finished_sims[0])
    output_sim.pars['parallelized'] = n # Store how this was parallelized
    output_sim.pars['n'] *= n # Restore this since used in later calculations -- a bit hacky, it's true
    
    for sim in finished_sims[1:]: # Skip the first one
        output_sim.people.update(sim.people)
        for key,val in sim.results.items():
            if key != 't':
                output_sim.results[key] += sim.results[key]
    
    return output_sim
    
    
    