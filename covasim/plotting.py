'''
Core plotting functions for simulations, multisims, and scenarios.

Also includes Plotly-based plotting functions to supplement the Matplotlib based
ones that are of the Sim and Scenarios objects. Intended mostly for use with the
webapp.
'''

import numpy as np
import pylab as pl
import sciris as sc
import datetime as dt
import matplotlib.ticker as ticker
from . import misc as cvm
from . import defaults as cvd
from . import settings as cvset


__all__ = ['date_formatter', 'plot_sim', 'plot_scens', 'plot_result', 'plot_compare', 'plot_people', 'plotly_sim', 'plotly_people', 'plotly_animate']


#%% Plotting helper functions


def handle_args(fig_args=None, plot_args=None, scatter_args=None, axis_args=None, fill_args=None,
                legend_args=None, date_args=None, show_args=None, mpl_args=None, **kwargs):
    ''' Handle input arguments -- merge user input with defaults; see sim.plot for documentation '''

    # Set defaults
    defaults = sc.objdict()
    defaults.fig     = sc.objdict(figsize=(10, 8))
    defaults.plot    = sc.objdict(lw=1.5, alpha= 0.7)
    defaults.scatter = sc.objdict(s=20, marker='s', alpha=0.7, zorder=0)
    defaults.axis    = sc.objdict(left=0.10, bottom=0.08, right=0.95, top=0.95, wspace=0.30, hspace=0.30)
    defaults.fill    = sc.objdict(alpha=0.2)
    defaults.legend  = sc.objdict(loc='best', frameon=False)
    defaults.date    = sc.objdict(as_dates=True, dateformat=None, interval=None, rotation=None, start_day=None, end_day=None)
    defaults.show    = sc.objdict(data=True, ticks=True, interventions=True, legend=True)
    defaults.mpl     = sc.objdict(dpi=None, fontsize=None, fontfamily=None) # Use Covasim global defaults

    # Handle directly supplied kwargs
    for dkey,default in defaults.items():
        keys = list(kwargs.keys())
        for kw in keys:
            if kw in default.keys():
                default[kw] = kwargs.pop(kw)

    # Merge arguments together
    args = sc.objdict()
    args.fig     = sc.mergedicts(defaults.fig,     fig_args)
    args.plot    = sc.mergedicts(defaults.plot,    plot_args)
    args.scatter = sc.mergedicts(defaults.scatter, scatter_args)
    args.axis    = sc.mergedicts(defaults.axis,    axis_args)
    args.fill    = sc.mergedicts(defaults.fill,    fill_args)
    args.legend  = sc.mergedicts(defaults.legend,  legend_args)
    args.date    = sc.mergedicts(defaults.date,    fill_args)
    args.show    = sc.mergedicts(defaults.show,    show_args)
    args.mpl     = sc.mergedicts(defaults.mpl,     mpl_args)

    # If unused keyword arguments remain, raise an error
    if len(kwargs):
        notfound = sc.strjoin(kwargs.keys())
        valid = sc.strjoin(sorted(set([k for d in defaults.values() for k in d.keys()]))) # Remove duplicates and order
        errormsg = f'The following keywords could not be processed:\n{notfound}\n\n'
        errormsg += f'Valid keywords are:\n{valid}\n\n'
        errormsg += 'For more precise plotting control, use fig_args, plot_args, etc.'
        raise sc.KeyNotFoundError(errormsg)

    # Handle what to show
    show_keys = defaults.show.keys()
    args.show = {k:True for k in show_keys}
    if show_args in [True, False]: # Handle all on or all off
        args.show = {k:show_args for k in show_keys}
    else:
        args.show = sc.mergedicts(args.show, show_args)

    # Handle global Matplotlib arguments
    args.mpl_orig = sc.objdict()
    for key,value in args.mpl.items():
        if value is not None:
            args.mpl_orig[key] = cvset.options.get(key)
            cvset.options.set(key, value)

    return args


def handle_to_plot(kind, to_plot, n_cols, sim, check_ready=True):
    ''' Handle which quantities to plot '''

    # Check that results are ready
    if check_ready and not sim.results_ready:
        errormsg = 'Cannot plot since results are not ready yet -- did you run the sim?'
        raise RuntimeError(errormsg)

    # If not specified or specified as a string, load defaults
    if to_plot is None or isinstance(to_plot, str):
        to_plot = cvd.get_default_plots(to_plot, kind=kind, sim=sim)

    # If a list of keys has been supplied
    if isinstance(to_plot, list):
        to_plot_list = to_plot # Store separately
        to_plot = sc.odict() # Create the dict
        reskeys = sim.result_keys()
        for reskey in to_plot_list:
            name = sim.results[reskey].name if reskey in reskeys else sim.results['strain'][reskey].name
            to_plot[name] = [reskey] # Use the result name as the key and the reskey as the value

    to_plot = sc.odict(sc.dcp(to_plot)) # In case it's supplied as a dict

    # Handle rows and columns -- assume 5 is the most rows we would want
    n_plots = len(to_plot)
    if n_cols is None:
        max_rows = 4 # Assumption -- if desired, the user can override this by setting n_cols manually
        n_cols = int((n_plots-1)//max_rows + 1) # This gives 1 column for 1-4, 2 for 5-8, etc.
    n_rows,n_cols = sc.get_rows_cols(n_plots, ncols=n_cols) # Inconsistent naming due to Covasim/Matplotlib conventions

    return to_plot, n_cols, n_rows


def create_figs(args, sep_figs, fig=None, ax=None):
    '''
    Create the figures and set overall figure properties. If a figure is supplied,
    reset the axes labels for automatic use by other plotting functions (i.e. ax1, ax2, etc.)
    '''
    if sep_figs:
        fig = None
        figs = []
    else:
        if fig is None:
            if ax is None:
                fig = pl.figure(**args.fig) # Create the figure if none is supplied
            else:
                fig = ax.figure
        else:
            for i,fax in enumerate(fig.axes):
                fax.set_label(f'ax{i+1}')
        figs = None
    pl.subplots_adjust(**args.axis)
    return fig, figs


def create_subplots(figs, fig, shareax, n_rows, n_cols, pnum, fig_args, sep_figs, log_scale, title):
    ''' Create subplots and set logarithmic scale '''

    # Try to find axes by label, if they've already been defined -- this is to avoid the deprecation warning of reusing axes
    label = f'ax{pnum+1}'
    ax = None
    try:
        for fig_ax in fig.axes:
            if fig_ax.get_label() == label:
                ax = fig_ax
                break
    except:
        pass

    # Handle separate figs
    if sep_figs:
        figs.append(pl.figure(**fig_args))
        if ax is None:
            ax = pl.subplot(111, label=label)
    else:
        if ax is None:
            ax = pl.subplot(n_rows, n_cols, pnum+1, sharex=shareax, label=label)

    # Handle log scale
    if log_scale:
        if isinstance(log_scale, list):
            if title in log_scale:
                ax.set_yscale('log')
        else:
            ax.set_yscale('log')

    return ax


def plot_data(sim, ax, key, scatter_args, color=None):
    ''' Add data to the plot '''
    if sim.data is not None and key in sim.data and len(sim.data[key]):
        if color is None:
            color = sim.results[key].color
        data_t = (sim.data.index-sim['start_day'])/np.timedelta64(1,'D') # Convert from data date to model output index based on model start date
        ax.scatter(data_t, sim.data[key], c=[color], label='Data', **scatter_args)
    return


def plot_interventions(sim, ax):
    ''' Add interventions to the plot '''
    for intervention in sim['interventions']:
        if hasattr(intervention, 'plot_intervention'): # Don't plot e.g. functions
            intervention.plot_intervention(sim, ax)
    return


def title_grid_legend(ax, title, grid, commaticks, setylim, legend_args, show_legend=True):
    ''' Plot styling -- set the plot title, add a legend, and optionally add gridlines'''

    # Handle show_legend being in the legend args, since in some cases this is the only way it can get passed
    if 'show_legend' in legend_args:
        show_legend = legend_args.pop('show_legend')
        popped = True
    else:
        popped = False

    # Show the legend
    if show_legend:
        ax.legend(**legend_args)

    # If we removed it from the legend_args dict, put it back now
    if popped:
        legend_args['show_legend'] = show_legend

    # Set the title and gridlines
    ax.set_title(title)
    ax.grid(grid)

    # Set the y axis style
    if setylim:
        ax.set_ylim(bottom=0)
    if commaticks:
        ylims = ax.get_ylim()
        if ylims[1] >= 1000:
            sc.commaticks(ax=ax)

    return


def date_formatter(start_day=None, dateformat=None, interval=None, start=None, end=None, ax=None, sim=None):
    '''
    Create an automatic date formatter based on a number of days and a start day.

    Wrapper for Matplotlib's date formatter. Note, start_day is not required if the
    axis uses dates already. To be used in conjunction with setting the x-axis
    tick label formatter.

    Args:
        start_day (str/date): the start day, either as a string or date object
        dateformat (str): the date format (default '%b-%d')
        interval (int): if supplied, the interval between ticks (must supply an axis also to take effect)
        start (str/int): if supplied, the lower limit of the axis
        end (str/int): if supplied, the upper limit of the axis
        ax (axes): if supplied, automatically set the x-axis formatter for this axis
        sim (Sim): if supplied, get the start day from this

    **Examples**::

        # Automatically configure the axis with default option
        cv.date_formatter(sim=sim, ax=ax)

        # Manually configure
        ax = pl.subplot(111)
        ax.plot(np.arange(60), np.random.random(60))
        formatter = cv.date_formatter(start_day='2020-04-04', interval=7, start='2020-05-01', end=50, dateformat='%Y-%m-%d', ax=ax)
        ax.xaxis.set_major_formatter(formatter)
    '''

    # Set the default -- "Mar-01"
    if dateformat is None:
        dateformat = '%b-%d'

    # Convert to a date object
    if start_day is None and sim is not None:
        start_day = sim['start_day']
    if start_day is None:
        errormsg = 'If not supplying a start day, you must supply a sim object'
        raise ValueError(errormsg)
    start_day = sc.date(start_day)

    @ticker.FuncFormatter
    def mpl_formatter(x, pos):
        return (start_day + dt.timedelta(days=int(x))).strftime(dateformat)

    # Set initial tick marks (intervals and limits)
    if ax is not None:

        # Handle limits
        xmin, xmax = ax.get_xlim()
        if start:
            xmin = sc.day(start, start_day=start_day)
        if end:
            xmax = sc.day(end, start_day=start_day)
        ax.set_xlim((xmin, xmax))

        # Set the x-axis intervals
        if interval:
            ax.set_xticks(np.arange(xmin, xmax+1, interval))

        # Set the formatter
        ax.xaxis.set_major_formatter(mpl_formatter)

    return mpl_formatter


def reset_ticks(ax, sim=None, date_args=None, start_day=None):
    ''' Set the tick marks, using dates by default '''

    # Handle options
    date_args = sc.objdict(date_args) # Ensure it's not a regular dict
    if start_day is None and sim is not None:
        start_day = sim['start_day']

    # Handle start and end days
    xmin,xmax = ax.get_xlim()
    if date_args.start_day:
        xmin = float(sc.day(date_args.start_day, start_day=start_day)) # Keep original type (float)
    if date_args.end_day:
        xmax = float(sc.day(date_args.end_day, start_day=start_day))
    ax.set_xlim([xmin, xmax])

    # Set the x-axis intervals
    if date_args.interval:
        ax.set_xticks(np.arange(xmin, xmax+1, date_args.interval))

    # Set xticks as dates
    if date_args.as_dates:

        date_formatter(start_day=start_day, dateformat=date_args.dateformat, ax=ax)
        if not date_args.interval:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Handle rotation
    if date_args.rotation:
        ax.tick_params(axis='x', labelrotation=date_args.rotation)

    return


def tidy_up(fig, figs, sep_figs, do_save, fig_path, do_show, args):
    ''' Handle saving, figure showing, and what value to return '''

    # Handle saving
    if do_save:
        if fig_path is not None: # No figpath provided - see whether do_save is a figpath
            fig_path = sc.makefilepath(fig_path) # Ensure it's valid, including creating the folder
        cvm.savefig(filename=fig_path) # Save the figure

    # Show the figure, or close it
    do_show = cvset.handle_show(do_show)
    if cvset.options.close and not do_show:
        if sep_figs:
            for fig in figs:
                pl.close(fig)
        else:
            pl.close(fig)

    # Reset Matplotlib defaults
    for key,value in args.mpl_orig.items():
        cvset.options.set(key, value)

    # Return the figure or figures
    if sep_figs:
        return figs
    else:
        return fig


def set_line_options(input_args, reskey, resnum, default):
    '''From the supplied line argument, usually a color or label, decide what to use '''
    if input_args is not None:
        if isinstance(input_args, dict): # If it's a dict, pull out this value
            output = input_args[reskey]
        elif isinstance(input_args, list): # If it's a list, ditto
            output = input_args[resnum]
        else: # Otherwise, assume it's the same value for all
            output = input_args
    else:
        output = default # Default value
    return output



#%% Core plotting functions

def plot_sim(to_plot=None, sim=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
         scatter_args=None, axis_args=None, fill_args=None, legend_args=None, date_args=None,
         show_args=None, mpl_args=None, n_cols=None, grid=False, commaticks=True,
         setylim=True, log_scale=False, colors=None, labels=None, do_show=None, sep_figs=False,
         fig=None, ax=None, **kwargs):
    ''' Plot the results of a single simulation -- see Sim.plot() for documentation '''

    # Handle inputs
    args = handle_args(fig_args=fig_args, plot_args=plot_args, scatter_args=scatter_args, axis_args=axis_args, fill_args=fill_args,
                       legend_args=legend_args, show_args=show_args, date_args=date_args, mpl_args=mpl_args, **kwargs)
    to_plot, n_cols, n_rows = handle_to_plot('sim', to_plot, n_cols, sim=sim)
    fig, figs = create_figs(args, sep_figs, fig, ax)

    # Do the plotting
    strain_keys = sim.result_keys('strain')
    for pnum,title,keylabels in to_plot.enumitems():
        ax = create_subplots(figs, fig, ax, n_rows, n_cols, pnum, args.fig, sep_figs, log_scale, title)
        for resnum,reskey in enumerate(keylabels):
            res_t = sim.results['t']
            if reskey in strain_keys:
                res = sim.results['strain'][reskey]
                ns = sim['n_strains']
                strain_colors = sc.gridcolors(ns)
                for strain in range(ns):
                    color = strain_colors[strain]  # Choose the color
                    label = 'wild type' if strain == 0 else sim['strains'][strain-1].label
                    if res.low is not None and res.high is not None:
                        ax.fill_between(res_t, res.low[strain,:], res.high[strain,:], color=color, **args.fill)  # Create the uncertainty bound
                    ax.plot(res_t, res.values[strain,:], label=label, **args.plot, c=color)  # Actually plot the sim!
            else:
                res = sim.results[reskey]
                color = set_line_options(colors, reskey, resnum, res.color)  # Choose the color
                label = set_line_options(labels, reskey, resnum, res.name)  # Choose the label
                if res.low is not None and res.high is not None:
                    ax.fill_between(res_t, res.low, res.high, color=color, **args.fill)  # Create the uncertainty bound
                ax.plot(res_t, res.values, label=label, **args.plot, c=color)  # Actually plot the sim!
            if args.show['data']:
                plot_data(sim, ax, reskey, args.scatter, color=color)  # Plot the data
            if args.show['ticks']:
                reset_ticks(ax, sim, args.date) # Optionally reset tick marks (useful for e.g. plotting weeks/months)
        if args.show['interventions']:
            plot_interventions(sim, ax) # Plot the interventions
        if args.show['legend']:
            title_grid_legend(ax, title, grid, commaticks, setylim, args.legend) # Configure the title, grid, and legend

    return tidy_up(fig, figs, sep_figs, do_save, fig_path, do_show, args)


def plot_scens(to_plot=None, scens=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
         scatter_args=None, axis_args=None, fill_args=None, legend_args=None, date_args=None,
         show_args=None, mpl_args=None, n_cols=None, grid=False, commaticks=True, setylim=True,
         log_scale=False, colors=None, labels=None, do_show=None, sep_figs=False, fig=None, ax=None, **kwargs):
    ''' Plot the results of a scenario -- see Scenarios.plot() for documentation '''

    # Handle inputs
    args = handle_args(fig_args=fig_args, plot_args=plot_args, scatter_args=scatter_args, axis_args=axis_args, fill_args=fill_args,
                   legend_args=legend_args, show_args=show_args, date_args=date_args, mpl_args=mpl_args, **kwargs)
    to_plot, n_cols, n_rows = handle_to_plot('scens', to_plot, n_cols, sim=scens.base_sim, check_ready=False) # Since this sim isn't run
    fig, figs = create_figs(args, sep_figs, fig, ax)

    # Do the plotting
    default_colors = sc.gridcolors(ncolors=len(scens.sims))
    for pnum,title,reskeys in to_plot.enumitems():
        ax = create_subplots(figs, fig, ax, n_rows, n_cols, pnum, args.fig, sep_figs, log_scale, title)
        reskeys = sc.promotetolist(reskeys) # In case it's a string
        for reskey in reskeys:
            resdata = scens.results[reskey]
            for snum,scenkey,scendata in resdata.enumitems():
                sim = scens.sims[scenkey][0] # Pull out the first sim in the list for this scenario
                strain_keys = sim.result_keys('strain')
                if reskey in strain_keys:
                    ns = sim['n_strains']
                    strain_colors = sc.gridcolors(ns)
                    for strain in range(ns):
                        res_y = scendata.best[strain,:]
                        color = strain_colors[strain]  # Choose the color
                        label = 'wild type' if strain == 0 else sim['strains'][strain - 1].label
                        ax.fill_between(scens.tvec, scendata.low[strain,:], scendata.high[strain,:], color=color, **args.fill)  # Create the uncertainty bound
                        ax.plot(scens.tvec, res_y, label=label, c=color, **args.plot)  # Plot the actual line
                        if args.show['data']:
                            plot_data(sim, ax, reskey, args.scatter, color=color)  # Plot the data
                else:
                    res_y = scendata.best
                    color = set_line_options(colors, scenkey, snum, default_colors[snum])  # Choose the color
                    label = set_line_options(labels, scenkey, snum, scendata.name)  # Choose the label
                    ax.fill_between(scens.tvec, scendata.low, scendata.high, color=color, **args.fill)  # Create the uncertainty bound
                    ax.plot(scens.tvec, res_y, label=label, c=color, **args.plot)  # Plot the actual line
                    if args.show['data']:
                        plot_data(sim, ax, reskey, args.scatter, color=color)  # Plot the data

                if args.show['interventions']:
                    plot_interventions(sim, ax) # Plot the interventions
                if args.show['ticks']:
                    reset_ticks(ax, sim, args.date) # Optionally reset tick marks (useful for e.g. plotting weeks/months)
        if args.show['legend']:
            title_grid_legend(ax, title, grid, commaticks, setylim, args.legend, pnum==0) # Configure the title, grid, and legend -- only show legend for first

    return tidy_up(fig, figs, sep_figs, do_save, fig_path, do_show, args)


def plot_result(key, sim=None, fig_args=None, plot_args=None, axis_args=None, scatter_args=None,
                date_args=None, mpl_args=None, grid=False, commaticks=True, setylim=True, color=None, label=None,
                do_show=None, do_save=False, fig_path=None, fig=None, ax=None, **kwargs):
    ''' Plot a single result -- see Sim.plot_result() for documentation '''

    # Handle inputs
    sep_figs = False # Only one figure
    fig_args  = sc.mergedicts({'figsize':(8,5)}, fig_args)
    axis_args = sc.mergedicts({'top': 0.95}, axis_args)
    args = handle_args(fig_args=fig_args, plot_args=plot_args, scatter_args=scatter_args, axis_args=axis_args,
                       date_args=date_args, mpl_args=mpl_args, **kwargs)
    fig, figs = create_figs(args, sep_figs, fig, ax)

    # Gather results
    res = sim.results[key]
    res_t = sim.results['t']
    if color is None:
        color = res.color

    # Reuse the figure, if available
    if ax is None: # Otherwise, make a new one
        try:
            ax = fig.axes[0]
        except:
            ax = fig.add_subplot(111, label='ax1')

    # Do the plotting
    if label is None:
        label = res.name
    if res.low is not None and res.high is not None:
        ax.fill_between(res_t, res.low, res.high, color=color, **args.fill) # Create the uncertainty bound
    ax.plot(res_t, res.values, c=color, label=label, **args.plot)
    plot_data(sim, ax, key, args.scatter, color=color) # Plot the data
    plot_interventions(sim, ax) # Plot the interventions
    title_grid_legend(ax, res.name, grid, commaticks, setylim, args.legend) # Configure the title, grid, and legend
    reset_ticks(ax, sim, args.date) # Optionally reset tick marks (useful for e.g. plotting weeks/months)

    return tidy_up(fig, figs, sep_figs, do_save, fig_path, do_show, args)


def plot_compare(df, log_scale=True, fig_args=None, axis_args=None, mpl_args=None, grid=False,
                 commaticks=True, setylim=True, color=None, label=None, fig=None, **kwargs):
    ''' Plot a MultiSim comparison -- see MultiSim.plot_compare() for documentation '''

    # Handle inputs
    fig_args  = sc.mergedicts({'figsize':(8,8)}, fig_args)
    axis_args = sc.mergedicts({'left': 0.16, 'bottom': 0.05, 'right': 0.98, 'top': 0.98, 'wspace': 0.50, 'hspace': 0.10}, axis_args)
    args = handle_args(fig_args=fig_args, axis_args=axis_args, mpl_args=mpl_args, **kwargs)
    fig, figs = create_figs(args, sep_figs=False, fig=fig)

    # Map from results into different categories
    mapping = {
        'cum': 'Cumulative counts',
        'new': 'New counts',
        'n': 'Number in state',
        'r': 'R_eff',
        }
    category = []
    for v in df.index.values:
        v_type = v.split('_')[0]
        if v_type in mapping:
            category.append(v_type)
        else:
            category.append('other')
    df['category'] = category

    # Plot
    for i,m in enumerate(mapping):
        not_r_eff = m != 'r'
        if not_r_eff:
            ax = fig.add_subplot(2, 2, i+1)
        else:
            ax = fig.add_subplot(8, 2, 10)
        dfm = df[df['category'] == m]
        logx = not_r_eff and log_scale
        dfm.plot(ax=ax, kind='barh', logx=logx, legend=False)
        if not(not_r_eff):
            ax.legend(loc='upper left', bbox_to_anchor=(0,-0.3))
        ax.grid(True)

    return fig


#%% Other plotting functions
def plot_people(people, bins=None, width=1.0, alpha=0.6, fig_args=None, axis_args=None,
                plot_args=None, do_show=None, fig=None):
    ''' Plot statistics of a population -- see People.plot() for documentation '''

    # Handle inputs
    if bins is None:
        bins = np.arange(0,101)

    # Set defaults
    color     = [0.1,0.1,0.1] # Color for the age distribution
    n_rows    = 4 # Number of rows of plots
    offset    = 0.5 # For ensuring the full bars show up
    gridspace = 10 # Spacing of gridlines
    zorder    = 10 # So plots appear on top of gridlines

    # Handle other arguments
    fig_args  = sc.mergedicts(dict(figsize=(18,11)), fig_args)
    axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.35), axis_args)
    plot_args = sc.mergedicts(dict(lw=1.5, alpha=0.6, c=color, zorder=10), plot_args)

    # Compute statistics
    min_age = min(bins)
    max_age = max(bins)
    edges = np.append(bins, np.inf) # Add an extra bin to end to turn them into edges
    age_counts = np.histogram(people.age, edges)[0]

    # Create the figure
    if fig is None:
        fig = pl.figure(**fig_args)
    pl.subplots_adjust(**axis_args)

    # Plot age histogram
    pl.subplot(n_rows,2,1)
    pl.bar(bins, age_counts, color=color, alpha=alpha, width=width, zorder=zorder)
    pl.xlim([min_age-offset,max_age+offset])
    pl.xticks(np.arange(0, max_age+1, gridspace))
    pl.grid(True)
    pl.xlabel('Age')
    pl.ylabel('Number of people')
    pl.title(f'Age distribution ({len(people):n} people total)')

    # Plot cumulative distribution
    pl.subplot(n_rows,2,2)
    age_sorted = sorted(people.age)
    y = np.linspace(0, 100, len(age_sorted)) # Percentage, not hard-coded!
    pl.plot(age_sorted, y, '-', **plot_args)
    pl.xlim([0,max_age])
    pl.ylim([0,100]) # Percentage
    pl.xticks(np.arange(0, max_age+1, gridspace))
    pl.yticks(np.arange(0, 101, gridspace)) # Percentage
    pl.grid(True)
    pl.xlabel('Age')
    pl.ylabel('Cumulative proportion (%)')
    pl.title(f'Cumulative age distribution (mean age: {people.age.mean():0.2f} years)')

    # Calculate contacts
    lkeys = people.layer_keys()
    n_layers = len(lkeys)
    contact_counts = sc.objdict()
    for lk in lkeys:
        layer = people.contacts[lk]
        p1ages = people.age[layer['p1']]
        p2ages = people.age[layer['p2']]
        contact_counts[lk] = np.histogram(p1ages, edges)[0] + np.histogram(p2ages, edges)[0]

    # Plot contacts
    layer_colors = sc.gridcolors(n_layers)
    share_ax = None
    for w,w_type in enumerate(['total', 'percapita', 'weighted']): # Plot contacts in different ways
        for i,lk in enumerate(lkeys):
            if w_type == 'total':
                weight = 1
                total_contacts = 2*len(people.contacts[lk]) # x2 since each contact is undirected
                ylabel = 'Number of contacts'
                title = f'Total contacts for layer "{lk}": {total_contacts:n}'
            elif w_type == 'percapita':
                weight = np.divide(1.0, age_counts, where=age_counts>0)
                mean_contacts = 2*len(people.contacts[lk])/len(people) # Factor of 2 since edges are bi-directional
                ylabel = 'Per capita number of contacts'
                title = f'Mean contacts for layer "{lk}": {mean_contacts:0.2f}'
            elif w_type == 'weighted':
                weight = people.pars['beta_layer'][lk]*people.pars['beta']
                total_weight = np.round(weight*2*len(people.contacts[lk]))
                ylabel = 'Weighted number of contacts'
                title = f'Total weight for layer "{lk}": {total_weight:n}'

            ax = pl.subplot(n_rows, n_layers, n_layers*(w+1)+i+1, sharey=share_ax)
            pl.bar(bins, contact_counts[lk]*weight, color=layer_colors[i], width=width, zorder=zorder, alpha=alpha)
            pl.xlim([min_age-offset,max_age+offset])
            pl.xticks(np.arange(0, max_age+1, gridspace))
            pl.grid(True)
            pl.xlabel('Age')
            pl.ylabel(ylabel)
            pl.title(title)
            if w_type == 'weighted':
                share_ax = ax # Update shared axis

    cvset.handle_show(do_show)

    return fig


#%% Plotly functions

def import_plotly():
    ''' Try to import Plotly, but fail quietly if not available '''

    # Try to import Plotly normally
    try:
        import plotly.graph_objects as go
        return go

    # If that failed, handle it gracefully
    except Exception as E:

        class PlotlyImportFailed(object):
            ''' Define a micro-class to give a helpful error message if the import failed '''

            def __init__(self, E):
                self.E = E

            def __getattr__(self, attr):
                errormsg = f'Plotly import failed: {str(self.E)}. Plotly plotting is not available. Please install Plotly first.'
                raise ImportError(errormsg)

        go = PlotlyImportFailed(E)
        return go


def get_individual_states(sim):
    ''' Helper function to convert people into integers '''

    people = sim.people

    states = [
        {'name': 'Healthy',
         'quantity': None,
         'color': '#a6cee3',
         'value': 0
         },
        {'name': 'Exposed',
         'quantity': 'date_exposed',
         'color': '#ff7f00',
         'value': 2
         },
        {'name': 'Infectious',
         'quantity': 'date_infectious',
         'color': '#e33d3e',
         'value': 3
         },
        {'name': 'Recovered',
         'quantity': 'date_recovered',
         'color': '#3e89bc',
         'value': 4
         },
        {'name': 'Dead',
         'quantity': 'date_dead',
         'color': '#000000',
         'value': 5
         },
    ]

    z = np.zeros((len(people), sim.npts))
    for state in states:
        date = state['quantity']
        if date is not None:
            inds = sim.people.defined(date)
            for ind in inds:
                z[ind, int(people[date][ind]):] = state['value']

    return z, states


# Default settings for the Plotly legend
plotly_legend = dict(legend_orientation="h", legend=dict(x=0.0, y=1.18))


def plotly_interventions(sim, fig, add_to_legend=False):
    ''' Add vertical lines for interventions to the plot '''
    go = import_plotly() # Load Plotly
    if sim['interventions']:
        for interv in sim['interventions']:
            if hasattr(interv, 'days'):
                for interv_day in interv.days:
                    if interv_day and interv_day < sim['n_days']:
                        interv_date = sim.date(interv_day, as_date=True)
                        fig.add_shape(dict(type="line", xref="x", yref="paper", x0=interv_date, x1=interv_date, y0=0, y1=1, line=dict(width=0.5, dash='dash')))
                        if add_to_legend:
                            fig.add_trace(go.Scatter(x=[interv_date], y=[0], mode='lines', name='Intervention change', line=dict(width=0.5, dash='dash')))
    return


def plotly_sim(sim, do_show=False):
    ''' Main simulation results -- parallel of sim.plot() '''

    go = import_plotly() # Load Plotly
    plots = []
    to_plot = cvd.get_default_plots()
    for p,title,keylabels in to_plot.enumitems():
        fig = go.Figure()
        for key in keylabels:
            label = sim.results[key].name
            this_color = sim.results[key].color
            x = sim.results['date'][:]
            y = sim.results[key][:]
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=label, line_color=this_color))
            if sim.data is not None and key in sim.data:
                xdata = sim.data['date']
                ydata = sim.data[key]
                fig.add_trace(go.Scatter(x=xdata, y=ydata, mode='markers', name=label + ' (data)', line_color=this_color))

        plotly_interventions(sim, fig, add_to_legend=(p==0)) # Only add the intervention label to the legend for the first plot
        fig.update_layout(title={'text':title}, yaxis_title='Count', autosize=True, **plotly_legend)

        plots.append(fig)

    if do_show:
        for fig in plots:
            fig.show()

    return plots


def plotly_people(sim, do_show=False):
    ''' Plot a "cascade" of people moving through different states '''

    go = import_plotly() # Load Plotly
    z, states = get_individual_states(sim)
    fig = go.Figure()

    for state in states[::-1]:  # Reverse order for plotting
        x = sim.results['date'][:]
        y = (z == state['value']).sum(axis=0)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            stackgroup='one',
            line=dict(width=0.5, color=state['color']),
            fillcolor=state['color'],
            hoverinfo="y+name",
            name=state['name']
        ))

    plotly_interventions(sim, fig)
    fig.update_layout(yaxis_range=(0, sim.n))
    fig.update_layout(title={'text': 'Numbers of people by health state'}, yaxis_title='People', autosize=True, **plotly_legend)

    if do_show:
        fig.show()

    return fig


def plotly_animate(sim, do_show=False):
    ''' Plot an animation of each person in the sim '''

    go = import_plotly() # Load Plotly
    z, states = get_individual_states(sim)

    min_color = min(states, key=lambda x: x['value'])['value']
    max_color = max(states, key=lambda x: x['value'])['value']
    colorscale = [[x['value'] / max_color, x['color']] for x in states]

    aspect = 5
    y_size = int(np.ceil((z.shape[0] / aspect) ** 0.5))
    x_size = int(np.ceil(aspect * y_size))

    z = np.pad(z, ((0, x_size * y_size - z.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)

    days = sim.tvec

    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 200, "redraw": True},
                                    "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 16},
            "prefix": "Day: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 200},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # make data
    fig_dict["data"] = [go.Heatmap(z=np.reshape(z[:, 0], (y_size, x_size)),
                                   zmin=min_color,
                                   zmax=max_color,
                                   colorscale=colorscale,
                                   showscale=False,
                                   )]

    for state in states:
        fig_dict["data"].append(go.Scatter(x=[None], y=[None], mode='markers',
                                           marker=dict(size=10, color=state['color']),
                                           showlegend=True, name=state['name']))

    # make frames
    for i, day in enumerate(days):
        frame = {"data": [go.Heatmap(z=np.reshape(z[:, i], (y_size, x_size)))],
                 "name": i}
        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [i],
            {"frame": {"duration": 5, "redraw": True},
             "mode": "immediate", }
        ],
            "label": i,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)

    fig.update_layout(
    autosize=True,
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            automargin=True,
            showgrid=False,
            showline=False,
            showticklabels=False,
        ),
    )


    fig.update_layout(title={'text': 'Epidemic over time'}, **plotly_legend)

    if do_show:
        fig.show()

    return fig
