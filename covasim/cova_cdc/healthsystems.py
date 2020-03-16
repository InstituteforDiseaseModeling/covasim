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


class HealthSystem(sc.prettobj):
    '''
    Class for storing, analyzing, a plotting health systems data.
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

    @staticmethod
    def check_blh(self, keylist):
        ''' Check if keys are best, low, high '''
        has_b = 'best' in keylist
        has_l = 'low' in keylist
        has_h = 'high' in keylist
        is_blh = (has_b and has_l and has_h)
        return is_blh


    def parse_data(self):
        '''
        Ensure the data object has the right structure. It should be:
            data[scenario][result]

        '''

        # Check first two levels are dicts
        D = self.data # Shortcut
        assert isinstance(D, dict), f'Data must be dict, but you supplied {type(D)}'
        self.scenkeys = list(D.keys())
        sk0 = self.scenkeys[0]
        assert isinstance(D[sk0], dict), f'The first key in the data must also be a dict, but you supplied {type(D[sk0])}'


        self.blhfirst = True
        keys = list(D[sk0].keys())

        if self.check_blh(keys):
        if 'best' in keys

        return


    def analyze(self):
        self.parse_data()

        return

    def plot(self):
        return


def run_healthsystem(doplot=True, dosave=False, *args, **kwargs):
    healthsystem = HealthSystem(*args, **kwargs)
    healthsystem.analyze()
    if doplot:
        healthsystem.plot()

