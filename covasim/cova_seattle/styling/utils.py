# To be merged with cova_base/utils.py when working

import os
import pylab as pl
import matplotlib as mpl

def set_plot_styles(family=None, font_size=None):
    ''' Load custom styles for plots '''
    

    if font_size is None:
        font_size = 18

    available = ['quicksand', 'roboto']
    if family is None:
        family = available[0] # Change default here
    if family not in available:
        raise ValueError(f'Font choice {family} not available; must be in {available}')

    cwd = os.path.abspath(os.path.dirname(__file__))
    fname = os.path.join(cwd, f'{family}.ttf')
    properties = mpl.font_manager.FontProperties(fname=fname)
    # pl.rcParams['font.family'] = properties.get_name()
    pl.rcParams['font.size'] = font_size
    return properties


def test_figure_styling():

    n = 30

    fig = pl.figure(figsize=(20,16))
    prop = set_plot_styles()

    pl.plot(pl.rand(n))
    pl.title('Example plot', fontproperties=prop, fontsize=30)

    return fig

fig = test_figure_styling()