'''
This file contains code for plotting styles to be used by Cova-SIM (formerly, Covid-ABM).

A simple, non-epidemic example.
'''

#%% Imports
import numpy as np # style for aliasing
import pylab as pyl
import sciris as sc
import os
import math
import copy
import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap # style for importing more than one function
from matplotlib.ticker import LogLocator, LogFormatter
import matplotlib.font_manager as font_manager
import functools

# Set user-specific configurations
username = os.path.split(os.path.expanduser('~'))[-1]
fontdirdict = {
    'dmistry': '/home/dmistry/Dropbox (IDM)/GoogleFonts',
    'cliffk': '/home/cliffk/idm/covid-19/GoogleFonts',
}
if username not in fontdirdict:
    fontdirdict[username] = os.path.expanduser(os.path.expanduser('~'),'Dropbox','GoogleFonts')


def plot(results, do_save=None, fig_args=None, plot_args=None, scatter_args=None, subplots_args=None, as_days=True, font_args=None, axis_args=None, color_theme=None, verbose=None):
    '''
    Example of how to plot the results -- can supply arguments for both the figure and the plots.

    Parameters
    ----------
    do_save : bool or str
        Whether or not to save the figure. If a string, save to that filename.

    fig_args : dict
        Dictionary of kwargs to be passed to plt.figure()

    plot_args : dict
        Dictionary of kwargs to be passed to plt.plot()

    scatter_args : dict
        Dictionary of kwargs to be passed to plt.plot() when using a scatter plot style

    subplots_args : dict
        Dictionary of kwargs to be passed to plt.subplots_adjust()

    as_days : bool
        Whether to plot the x-axis as days or time points

    font_size : int
        Fontsize baseline

    font_args : dict
        Dictionary of font style args and paths

    axis_args : dict
        Dictionary of axis style args

    Returns
    -------
    Figure handle 
    '''

    # if verbose is None:
        # verbose = self['verbose']
    if verbose:
        print('Plotting...')

    if fig_args     is None: fig_args     = {'figsize':(14,11)}
    if plot_args    is None: plot_args    = {'lw':3, 'alpha':0.7}
    if scatter_args is None: scatter_args = {'s':150, 'marker':'s'}
    if subplots_args    is None: subplots_args    = {'left':0.12, 'bottom':0.08, 'right':0.95, 'top':0.94, 'wspace':0.2, 'hspace':0.4}


    if axis_args is None:
        if as_days:
            axis_args = {'xaxis_units': 4}

    if font_args is None:
        try:
            fontpath = fontdirdict[username]
            font_style = 'Roboto_Condensed'
            fontstyle_path = os.path.join(fontpath,font_style,font_style.replace('_','') + '-Light.ttf')
            prop = font_manager.FontProperties(fname = fontstyle_path)
            mplt.rcParams['font.family'] = prop.get_name()
        except:
            print("User doesn't have access to GoogleFonts folder...")
            pass
        font_args = {'font_size': 18}

    fig = plt.figure(**fig_args)
    plt.subplots_adjust(**subplots_args)
    plt.rcParams['font.size'] = font_args['font_size']

    res = results # Shorten since heavily used

    to_plot = sc.odict({ # TODO
        'Total counts': sc.odict({'n_susceptible':'Number susceptible', 
                              'n_exposed':'Number exposed', 
                              'n_infectious':'Number infectious',
                              'cum_diagnosed':'Number diagnosed',
                              'n_recoveries': 'Number of recoveries',
                            }),
        'Daily counts': sc.odict({'infections':'New infections',
                              'tests':'Number of tests',
                              'diagnoses':'New diagnoses', 
                              'deaths': 'Number of deaths',
                             }),
        })

    # Plot everything
    colors = sc.gridcolors(7)

    # make sure plots for similar kinds of data match by color between subplots
    if color_theme is 'primaries':  
        keys = [item for subkey in [to_plot[k].keys() for k in to_plot] for item in subkey]
        colors = dict.fromkeys(keys)
        colors['infections'] = '#d32828'
        colors['infectious'] = colors['infections']
        colors['deaths'] = '#111111'
        colors['recoveries'] = '#009342'
        colors['susceptible'] = '#1893e0'
        colors['tests'] = '#5500ad'
        colors['diagnoses'] = '#f38600'
        colors['diagnosed'] = colors['diagnoses']
        colors['exposed'] = '#ffd600'

    elif color_theme is 'muted_dark':
        keys = [item for subkey in [to_plot[k].keys() for k in to_plot] for item in subkey]
        colors = dict.fromkeys(keys)
        colors['red'] = '#cc1631'
        colors['orange'] = '#f97000'
        colors['green'] = '#7baf00'
        colors['blue'] = '#0186c4'
        # colors['yellow'] = '#edd821' # bright yellow
        colors['yellow'] = '#e8c412' # dark yellow
        colors['black'] = '#4c4c4c'
        colors['purple'] = '#4a04d8' # indigo

        colors['infections'] = colors['red'] 
        colors['infectious'] = colors['infections'] 
        colors['exposed'] = colors['orange']
        colors['susceptible'] = colors['blue']
        colors['diagnoses'] = colors['yellow']
        colors['diagnosed'] = colors['diagnoses']
        colors['recoveries'] = colors['green']
        colors['deaths'] = colors['black']
        colors['tests'] = colors['purple']

        # data_mapping = {
        #     'cum_diagnosed': pl.cumsum(self.data['new_positives']),
        #     'tests':         self.data['new_tests'],
        #     'diagnoses':     self.data['new_positives'],
        #     }

    for p,title,keylabels in to_plot.enumitems():
        plt.subplot(2,1,p+1)
        for i,key,label in keylabels.enumitems():
            if type(colors) == list:
                this_color = colors[p+i]
            elif type(colors) == dict:
                color_key = label.split(' ')[-1]
                this_color = colors[color_key]

            y = res[key]
            plt.plot(res['t'], y, label=label.title().replace(' Of ',' of '), **plot_args, color = this_color)

            # if key in data_mapping:
            #     pl.scatter(self.data['day'], data_mapping[key], c=[this_color], **scatter_args)
            # pl.scatter(pl.nan, pl.nan, c=[(0,0,0)], label='Data', **scatter_args)
            
        # just to see how everything looks together
        if p == 0:
            plt.plot(res['t'], res['tests'] * 1.2, color = colors['tests'], **plot_args)
            plt.plot(res['t'], res['deaths'], color = colors['deaths'], **plot_args)

        plt.grid(True)

        if as_days:
            max_xaxis = math.ceil(max(res['t'])/axis_args['xaxis_units']) * axis_args['xaxis_units']
            plt.xticks(np.arange(0,max_xaxis,axis_args['xaxis_units']))

        # cov_ut.fixaxis(self)
        plt.ylabel('Count')
        plt.xlabel('Days since index case')
        plt.title(title, fontsize = font_args['font_size'] + 4)
        plt.legend(loc=1,fontsize = font_args['font_size']-2)
        plt.tick_params(labelsize = font_args['font_size'])

    if do_save:
        if isinstance(do_save, str):
            filename = do_save
        else:
            if color_theme is None:
                filename = 'covid_abm_plotting_style_default.png'
            else:
                filename = 'covid_abm_plotting_style_' + color_theme + '.png'
        plt.savefig(filename)

    return fig


if __name__ == '__main__':

    to_plot = sc.odict({ # TODO
        'Total counts': sc.odict({'n_susceptible':'Number susceptible', 
                                  'n_exposed':'Number exposed', 
                                  'n_infectious':'Number infectious',
                                  'cum_diagnosed':'Number diagnosed',
                                }),
        'Daily counts': sc.odict({'infections':'New infections',
                                  'tests':'Number of tests',
                                  'diagnoses':'New diagnoses', 
                                 }),
        })

    n_time = 42
    keys = [item for subkey in [to_plot[k].keys() for k in to_plot] for item in subkey]
    res = dict.fromkeys(keys)
    # res['t'] = np.arange(n_time)
    res['t'] = np.arange(0,n_time,0.1)
    res['n_susceptible'] = np.sin(res['t'] + 3) * 800
    res['n_exposed'] = np.sin(res['t'] + 6) * 800
    res['n_infectious'] = np.sin(res['t'] + 9) * 200
    res['infections'] = np.cos(res['t'] + 2) * 200
    res['tests'] =  np.sin(2 * res['t']) * 200
    res['diagnoses'] = np.cos(res['t']) * 80
    res['cum_diagnosed'] = np.cumsum(res['diagnoses']) * 4
    res['deaths'] = np.cumsum(res['diagnoses']) * 0.2
    res['n_recoveries'] = np.cos(res['t']) * 1600

    plot(res,do_save=True,as_days=True,axis_args={'xaxis_units': 10}, color_theme='primaries')
