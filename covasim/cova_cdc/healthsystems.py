'''
Perform modeling of health systems capacity.
'''

import sciris as sc

__all__ = ['make_hspars', 'HealthSystem']

def make_hspars():
    ''' Make defaults for health system parameters '''
    hspars = {}
    return hspars


class HealthSystem(sc.prettobj):

    def __init__(self, data=None, filename=None):
        if filename is not None:
            data = sc.loadobj(filename)
        self.data = data
        self.results = None
        return

    def analyze(self):
        return

    def plot(self):
        return

