'''
Set the parameters for COVID-ABM -- placeholder for actual functions in applications
folders.
'''


__all__ = ['make_pars', 'get_age_sex', 'load_data']


def make_pars():
    ''' Set parameters for the simulation '''
    raise NotImplementedError


def get_age_sex():
    ''' Get the age and sex of each person '''
    raise NotImplementedError


def load_data(filename=None):
    ''' Load data for comparing to the model output '''
    raise NotImplementedError



