'''
Perform modeling of health systems capacity.
'''

import pylab as pl
import sciris as sc

__all__ = ['make_hspars', 'HealthSystem']


def make_hspars():
    '''
    Make defaults for health system parameters. Estimates from:
        https://docs.google.com/document/d/1fIs2kCuu33tTCpbHQ-0YfvqXP4bSrqg-/edit#heading=h.gjdgxs
    '''
    hspars = sc.objdict()
    hspars.symptomatic  = 0.5 # Fraction of cases that are symptomatic
    hspars.hospitalized = 0.25 # Fraction of sympotmatic cases that require hospitalization
    hspars.icu          = 0.08 # Fraction of symptomatic cases that require ICU
    hspars.moderate_dur = pl.mean([7.9, 13.4]) # Days of a moderate stay
    hspars.severe_dur   = pl.mean([12.5, 21.2]) # Days of a severe stay
    hspars.acc_frac     = 0.5 # Fraction of time in severe cases that stay in an AAC bed

    return hspars


class HealthSystem(sc.prettyobj):
    '''
    Class for storing, analyzing, a plotting health systems data.

    Data are assumed to come from COVASim and be of the format:
        data[result_type][scenario_name][best,low,high] = time series
        e.g.
        data['cum_exposed']['baseline']['best'] = [0, 1, 1, 2, 3, 5, 10, 13 ...]
    '''

    def __init__(self, data=None, filename=None, hspars=None):
        if filename is not None:
            if data is not None:
                raise ValueError(f'You can supply data or a filename, but what am I supposed to do with both?')
            data = sc.loadobj(filename)
        if hspars is None:
            hspars = make_hspars()
        self.data = data
        self.hspars = hspars
        self.results = None
        return


    def parse_data(self):
        '''
        Ensure the data object has the right structure, and store the keys in the object.
        '''

        # Check first two three are dicts
        D = self.data # Shortcut
        if not isinstance(D, dict):
            raise TypeError(f'Data must be dict with keys for different results, but you supplied {type(D)}')

        self.reskeys = list(D.keys())
        rk0 = self.reskeys[0] # For "results key 0"
        if not isinstance(D[rk0], dict):
            raise TypeError(f'The second level in the data must also be a dict, but you supplied {type(D[rk0])}')

        self.scenkeys = list(D[rk0].keys())
        sk0 = self.scenkeys[0]
        if not isinstance(D[rk0][sk0], dict):
            raise TypeError(f'The third level in the data must also be a dict, but you supplied {type(D[rk0][sk0])}')

        self.blh = ['best', 'low', 'high']
        if not all([(key in D[rk0][sk0]) for key in self.blh]):
            raise ValueError(f'The required keys {self.blh} could not be found in {D[rk0][sk0].keys()}')
        if not sc.isinstance(D[rk0][sk0].best, 'arraylike'):
            raise TypeError(f'Was expecting a numeric array, but got {type(D[rk0][sk0].best)}')

        return


    def analyze(self):
        ''' Analyze the data and project resource needs '''
        hs = self.hspars # Shorten since used a lot
        self.parse_data() # Make sure the data has the right structure


        return

    def plot(self):
        return


def run_healthsystem(doplot=True, dosave=False, *args, **kwargs):
    healthsystem = HealthSystem(*args, **kwargs)
    healthsystem.analyze()
    if doplot:
        healthsystem.plot()

