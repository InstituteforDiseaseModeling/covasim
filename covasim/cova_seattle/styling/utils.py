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



def set_plot_styles(font_family=None, font_size=None, verbose=False):
    ''' Load custom styles for plots '''

    if font_family is None:
        font_families = ['Roboto', 'Quicksand', 'Proxima Nova']
    else:
        font_families = sc.promotetolist(font_family)

    # Find available system fonts
    # flist = mpl.font_manager.get_fontconfig_fonts()
    # available_fonts = []
    # for fname in flist:
    #     try:
    #         name = mpl.font_manager.FontProperties(fname=fname).get_name()
    #         available_fonts.append(name)
    #     except Exception as E:
    #         if verbose:
    #             print(f'Note: could not access system font {fname} ({str(E)})')

    # family = None
    # for name in font_families:
    #     if name in available_fonts:
    #         family = name
    #         break
    #     else:
    #         print(f'Note: font {name} not found, falling back to next choice')

    if font_size is None:
        font_size = 18

    family = font_families[0]

    # Note: loading a specific font doesn't work globally
    # cwd = os.path.abspath(os.path.dirname(__file__))
    # fname = os.path.join(cwd, f'{family}.ttf')
    # properties = mpl.font_manager.FontProperties(fname=fname)

    print('hi')
    pl.rcParams['font.family'] = family
    pl.rcParams['font.size'] = font_size
    print('ok')
    return


def test_figure_styling():

    n = 30

    fig = pl.figure(figsize=(20,16))
    cova.set_plot_styles(verbose=True)

    pl.plot(pl.rand(n))
    pl.title('Example plot')

    return fig

fig = test_figure_styling()