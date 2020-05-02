'''
Plotly-based plotting functions to supplement the Matplotlib based ones that are
part of the Sim and Scenarios objects. Intended mostly for use with the webapp.
'''

import numpy as np
import sciris as sc
import pylab as pl
import datetime as dt
import matplotlib.ticker as ticker
import plotly.graph_objects as go
from . import defaults as cvd


__all__ = ['plot_sim', 'plot_scens', 'plot_result', 'plotly_sim', 'plotly_people', 'plotly_animate']





def handle_args(which, to_plot, n_cols, fig_args, plot_args, scatter_args, axis_args, fill_args, legend_args):
    ''' Handle input arguments -- merge user input with defaults '''
    args = sc.objdict()
    args.fig     = sc.mergedicts({'figsize': (16, 14)}, fig_args)
    args.plot    = sc.mergedicts({'lw': 3, 'alpha': 0.7}, plot_args)
    args.scatter = sc.mergedicts({'s':70, 'marker':'s'}, scatter_args)
    args.axis    = sc.mergedicts({'left': 0.10, 'bottom': 0.05, 'right': 0.95, 'top': 0.97, 'wspace': 0.25, 'hspace': 0.25}, axis_args)
    args.fill    = sc.mergedicts({'alpha': 0.2}, fill_args)
    args.legend  = sc.mergedicts({'loc': 'best'}, legend_args)

    if to_plot is None:
        if which == 'sim':
            to_plot = cvd.get_sim_plots()
        elif which =='scens':
            to_plot = cvd.get_scen_plots()
        else:
            errormsg = f'Which must be "sim" or "scens", not {which}'
            raise NotImplementedError(errormsg)
    to_plot = sc.dcp(to_plot) # In case it's supplied as a dict

    n_rows = np.ceil(len(to_plot)/n_cols) # Number of subplot rows to have

    return to_plot, n_rows, args


def create_figs(sep_figs, args, font_size, font_family):
    if sep_figs:
        fig = None
        figs = []
    else:
        fig = pl.figure(**args.fig)
        figs = None
    pl.subplots_adjust(**args.axis)
    pl.rcParams['font.size'] = font_size
    if font_family:
        pl.rcParams['font.family'] = font_family
    return fig, figs, None # Initialize axis to be None


def create_subplots(figs, ax, n_rows, n_cols, rk, fig_args, sep_figs, log_scale, title):
    if sep_figs:
        figs.append(pl.figure(**fig_args))
        ax = pl.subplot(111)
    else:
        if rk == 0:
            ax = pl.subplot(n_rows, n_cols, rk+1)
        else:
            ax = pl.subplot(n_rows, n_cols, rk+1, sharex=ax)

    if log_scale:
        if isinstance(log_scale, list):
            if title in log_scale:
                ax.set_yscale('log')
        else:
            ax.set_yscale('log')

    return ax


def plot_data(sim, key, scatter_args):
    if sim.data is not None and key in sim.data and len(sim.data[key]):
        this_color = sim.results[key].color
        data_t = (sim.data.index-sim['start_day'])/np.timedelta64(1,'D') # Convert from data date to model output index based on model start date
        pl.scatter(data_t, sim.data[key], c=[this_color], **scatter_args)
        pl.scatter(pl.nan, pl.nan, c=[(0,0,0)], label='Data', **scatter_args)
    return


def plot_interventions(sim, ax): # Plot the interventions
    for intervention in sim['interventions']:
        intervention.plot(sim, ax)
    return


def title_grid_legend(title, grid, legend_args, show=True):
    pl.title(title)
    if show: # Only show the legend for some subplots
        pl.legend(**legend_args)
    pl.grid(grid)
    sc.setylim()
    return


def reset_ticks(ax, sim, commaticks, interval, as_dates):

    if commaticks:
        sc.commaticks()

    if interval:
        xmin,xmax = ax.get_xlim()
        ax.set_xticks(pl.arange(xmin, xmax+1, interval))

    # Set xticks as dates
    if as_dates:

        @ticker.FuncFormatter
        def date_formatter(x, pos):
            return (sim['start_day'] + dt.timedelta(days=x)).strftime('%b-%d')

        ax.xaxis.set_major_formatter(date_formatter)
        if not interval:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    return


def tidy_up(fig, figs, sep_figs, do_save, fig_path, do_show, default_name):
    if do_save:
        if fig_path is None: # No figpath provided - see whether do_save is a figpath
            fig_path = default_name # Just give it a default name
        fig_path = sc.makefilepath(fig_path) # Ensure it's valid, including creating the folder
        pl.savefig(fig_path)

    if do_show:
        pl.show()
    else:
        pl.close(fig)

    if sep_figs:
        return figs
    else:
        return fig


def plot_sim(sim, to_plot=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
         scatter_args=None, axis_args=None, fill_args=None, legend_args=None, as_dates=True, dateformat=None,
         interval=None, n_cols=1, font_size=18, font_family=None, grid=True, commaticks=True,
         log_scale=False, do_show=True, sep_figs=False, verbose=None):
    ''' Plot the results of a sim -- see Sim.plot() for documentation. '''

    # Handle inputs
    to_plot, n_rows, args = handle_args('sim', to_plot, n_cols, fig_args, plot_args, scatter_args, axis_args, fill_args, legend_args)
    fig, figs, ax = create_figs(sep_figs, args, font_size, font_family)

    # Do the plotting
    for pnum,title,keylabels in to_plot.enumitems():
        ax = create_subplots(figs, ax, n_rows, n_cols, pnum, args.fig, sep_figs, log_scale, title)
        for reskey in keylabels:
            res = sim.results[reskey]
            res_t = sim.results['t']
            pl.plot(res_t, res.values, label=res.name, **args.plot, c=res.color)
            plot_data(sim, reskey, args.scatter) # Plot the data
            plot_interventions(sim, ax) # Plot the interventions
            title_grid_legend(title, grid, args.legend) # Configure the title, grid, and legend
            reset_ticks(ax, sim, commaticks, interval, as_dates) # Optionally reset tick marks (useful for e.g. plotting weeks/months)

    return tidy_up(fig, figs, sep_figs, do_save, fig_path, do_show, default_name='covasim.png')


def plot_scens(scens, to_plot=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
         scatter_args=None, axis_args=None, fill_args=None, legend_args=None, as_dates=True, dateformat=None,
         interval=None, n_cols=1, font_size=18, font_family=None, grid=True, commaticks=True,
         log_scale=False, do_show=True, sep_figs=False):
    ''' Plot the results of a scenario -- see Scenarios.plot() for documentation. '''

    # Handle inputs
    to_plot, n_rows, args = handle_args('scens', to_plot, n_cols, fig_args, plot_args, scatter_args, axis_args, fill_args, legend_args)
    fig, figs, ax = create_figs(sep_figs, args, font_size, font_family)

    # Do the plotting
    for pnum,title,reskeys in to_plot.enumitems():
        ax = create_subplots(figs, ax, n_rows, n_cols, pnum, args.fig, sep_figs, log_scale, title)
        for reskey in reskeys:
            resdata = scens.results[reskey]
            for scenkey, scendata in resdata.items():
                sim = scens.sims[scenkey][0] # Pull out the first sim in the list for this scenario
                ax.fill_between(scens.tvec, scendata.low, scendata.high, **args.fill) # Create the uncertainty bound
                ax.plot(scens.tvec, scendata.best, label=scendata.name, **args.plot) # Plot the actual line
                plot_data(sim, reskey, args.scatter) # Plot the data
                plot_interventions(sim, ax) # Plot the interventions
                title_grid_legend(title, grid, args.legend, pnum==0) # Configure the title, grid, and legend -- only show legend for first
                reset_ticks(ax, sim, commaticks, interval, as_dates) # Optionally reset tick marks (useful for e.g. plotting weeks/months)

    return tidy_up(fig, figs, sep_figs, do_save, fig_path, do_show, default_name='covasim_scenarios.png')


def plot_result(sim, key, fig_args=None, plot_args=None):
    ''' Plot a single result -- see Sim.plot_result() for documentation. '''
    fig_args  = sc.mergedicts({'figsize':(16,10)}, fig_args)
    plot_args = sc.mergedicts({'lw':3, 'alpha':0.7}, plot_args)
    fig = pl.figure(**fig_args)
    pl.subplot(111)
    tvec = sim.results['t']
    res = sim.results[key]
    y = res.values
    color = res.color
    pl.plot(tvec, y, c=color, **plot_args)
    return fig


def get_individual_states(sim):
    ''' Helper function to convert people into integers '''

    people = sim.people

    states = [
        {'name': 'Healthy',
         'quantity': None,
         'color': '#b9d58a',
         'value': 0
         },
        {'name': 'Exposed',
         'quantity': 'date_exposed',
         'color': '#e37c30',
         'value': 2
         },
        {'name': 'Infectious',
         'quantity': 'date_infectious',
         'color': '#c35d86',
         'value': 3
         },
        {'name': 'Recovered',
         'quantity': 'date_recovered',
         'color': '#799956',
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


def plotly_sim(sim):
    ''' Main simulation results -- parallel of sim.plot() '''

    plots = []
    to_plot = cvd.get_sim_plots()
    for p,title,keylabels in to_plot.enumitems():
        fig = go.Figure()
        for key in keylabels:
            label = sim.results[key].name
            this_color = sim.results[key].color
            y = sim.results[key][:]
            fig.add_trace(go.Scatter(x=sim.results['t'][:], y=y, mode='lines', name=label, line_color=this_color))
            if sim.data is not None and key in sim.data:
                data_t = (sim.data.index-sim['start_day'])/np.timedelta64(1,'D')
                print(sim.data.index, sim['start_day'], np.timedelta64(1,'D'), data_t)
                ydata = sim.data[key]
                fig.add_trace(go.Scatter(x=data_t, y=ydata, mode='markers', name=label + ' (data)', line_color=this_color))

        if sim['interventions']:
            for interv in sim['interventions']:
                if hasattr(interv, 'days'):
                    for interv_day in interv.days:
                        if interv_day > 0 and interv_day < sim['n_days']:
                            fig.add_shape(dict(type="line", xref="x", yref="paper", x0=interv_day, x1=interv_day, y0=0, y1=1, name='Intervention', line=dict(width=0.5, dash='dash')))
                            fig.update_layout(annotations=[dict(x=interv_day, y=1.07, xref="x", yref="paper", text="Intervention change", showarrow=False)])

        fig.update_layout(title={'text':title}, xaxis_title='Day', yaxis_title='Count', autosize=True)

        plots.append(fig)
    return plots


def plotly_people(sim, do_show=False):
    ''' Plot a "cascade" of people moving through different states '''
    z, states = get_individual_states(sim)

    fig = go.Figure()

    for state in states[::-1]:  # Reverse order for plotting
        fig.add_trace(go.Scatter(
            x=sim.tvec, y=(z == state['value']).sum(axis=0),
            stackgroup='one',
            line=dict(width=0.5, color=state['color']),
            fillcolor=state['color'],
            hoverinfo="y+name",
            name=state['name']
        ))

    if sim['interventions']:
        for interv in sim['interventions']:
                if hasattr(interv, 'days'):
                    if interv.do_plot:
                        for interv_day in interv.days:
                            if interv_day > 0 and interv_day < sim['n_days']:
                                fig.add_shape(dict(type="line", xref="x", yref="paper", x0=interv_day, x1=interv_day, y0=0, y1=1, name='Intervention', line=dict(width=0.5, dash='dash')))
                                fig.update_layout(annotations=[dict(x=interv_day, y=1.07, xref="x", yref="paper", text="Intervention change", showarrow=False)])

    fig.update_layout(yaxis_range=(0, sim.n))
    fig.update_layout(title={'text': 'Numbers of people by health state'}, xaxis_title='Day', yaxis_title='People', autosize=True)

    if do_show:
        fig.show()

    return fig


def plotly_animate(sim, do_show=False):
    ''' Plot an animation of each person in the sim '''

    z, states = get_individual_states(sim)

    min_color = min(states, key=lambda x: x['value'])['value']
    max_color = max(states, key=lambda x: x['value'])['value']
    colorscale = [[x['value'] / max_color, x['color']] for x in states]

    aspect = 3
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
            "font": {"size": 20},
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
            automargin=True,
            range=[-0.5, x_size + 0.5],
            constrain="domain",
            showgrid=False,
            showline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            automargin=True,
            range=[-0.5, y_size + 0.5],
            constrain="domain",
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            showline=False,
            showticklabels=False,
        ),
    )


    fig.update_layout(title={'text': 'Epidemic over time'})

    if do_show:
        fig.show()

    return fig