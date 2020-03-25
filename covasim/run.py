'''
Functions for running multiple Covasim runs.
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import sciris as sc


# Specify all externally visible functions this file defines
__all__ = ['single_run', 'multi_run']


def single_run(sim, ind=0, noise=0.0, noisepar=None, verbose=None, sim_args=None, **kwargs):
    '''
    Convenience function to perform a single simulation run. Mostly used for
    parallelization, but can also be used directly:
        import covasim.cova_generic as cova
        sim = cova.Sim() # Create a default simulation
        sim = cova.single_run(sim) # Run it, equivalent(ish) to sim.run()
    '''

    if sim_args is None:
        sim_args = {}

    new_sim = sc.dcp(sim) # To avoid overwriting it; otherwise, use

    if verbose is None:
        verbose = new_sim['verbose']

    new_sim['seed'] += ind # Reset the seed, otherwise no point of parallel runs
    new_sim.set_seed()

    # If the noise parameter is not found, guess what it should be
    if noisepar is None:
        guesses = ['r_contact', 'r0', 'beta']
        found = [guess for guess in guesses if guess in sim.pars.keys()]
        if len(found)!=1:
            raise KeyError(f'Cound not guess noise parameter since out of {guesses}, {found} were found')
        else:
            noisepar = found[0]

    # Handle noise -- normally distributed fractional error
    noiseval = noise*np.random.normal()
    if noiseval > 0:
        noisefactor = 1 + noiseval
    else:
        noisefactor = 1/(1-noiseval)
    new_sim[noisepar] *= noisefactor

    if verbose>=1:
        print(f'Running a simulation using {new_sim["seed"]} seed and {noisefactor} noise')

    # Handle additional arguments
    for key,val in kwargs.items():
        print(f'Processing {key}:{val}')
        if key in new_sim.pars.keys():
            if verbose>=1:
                print(f'Setting key {key} from {new_sim[key]} to {val}')
                new_sim[key] = val
            pass
        else:
            raise KeyError(f'Could not set key {key}: not a valid parameter name')

    # Run
    new_sim.run(verbose=verbose)

    return new_sim


def multi_run(sim, n=4, noise=0.0, noisepar=None, iterpars=None, verbose=None, sim_args=None, combine=False, **kwargs):
    '''
    For running multiple runs in parallel. Example:
        import covid_seattle
        sim = covid_seattle.Sim()
        sims = covid_seattle.multi_run(sim, n=6, noise=0.2)
    '''

    # Create the sims
    if sim_args is None:
        sim_args = {}

    # Handle iterpars
    if iterpars is None:
        iterpars = {}
    else:
        n = None # Reset and get from length of dict instead
        for key,val in iterpars.items():
            new_n = len(val)
            if n is not None and new_n != n:
                raise ValueError(f'Each entry in iterpars must have the same length, not {n} and {len(val)}')
            else:
                n = new_n

    # Copy the simulations
    iterkwargs = {'ind':np.arange(n)}
    iterkwargs.update(iterpars)
    kwargs = {'sim':sim, 'noise':noise, 'noisepar':noisepar, 'verbose':verbose, 'sim_args':sim_args}
    sims = sc.parallelize(single_run, iterkwargs=iterkwargs, kwargs=kwargs)

    if not combine:
        output = sims
    else:
        print('WARNING: not tested!')
        output_sim = sc.dcp(sims[0])
        output_sim.pars['parallelized'] = n # Store how this was parallelized
        output_sim.pars['n'] *= n # Restore this since used in later calculations -- a bit hacky, it's true
        for sim in sims[1:]: # Skip the first one
            output_sim.people.update(sim.people)
            for key in sim.results_keys:
                output_sim.results[key] += sim.results[key]
        output = output_sim

    return output