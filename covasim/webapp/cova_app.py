'''
Sciris app to run the web interface.
'''

# Key imports
import covasim as cv
import os
import sys
import numpy as np
import plotly.graph_objects as go
import sciris as sc
import base64 # Download/upload-specific import
import json

# Check requirements, and if met, import scirisweb
cv.requirements.check_scirisweb(die=True)
import scirisweb as sw

# Create the app
app = sw.ScirisApp(__name__, name="Covasim")
app.sessions = dict() # For storing user data
flask_app = app.flask_app

#%% Define the API

# Set defaults
max_pop  = 10e3 # Maximum population size
max_days = 180  # Maximum number of days
max_time = 10   # Maximum of seconds for a run

@app.register_RPC()
def get_defaults(region=None, merge=False):
    ''' Get parameter defaults '''

    if region is None:
        region = 'Example'

    regions = {
        'scale': {
            'Example': 1,
            'Seattle': 25,
            # 'Wuhan': 200,
        },
        'n': {
            'Example': 2000,
            'Seattle': 10000,
            # 'Wuhan': 1,
        },
        'n_days': {
            'Example': 60,
            'Seattle': 45,
            # 'Wuhan': 90,
        },
        'n_infected': {
            'Example': 100,
            'Seattle': 4,
            # 'Wuhan': 10,
        },
        'web_int_day': {
            'Example': 25,
            'Seattle': 0,
            # 'Wuhan': 1,
        },
        'web_int_eff': {
            'Example': 0.8,
            'Seattle': 0.0,
            # 'Wuhan': 0.9,
        },
    }

    sim_pars = {}
    sim_pars['scale']       = dict(best=1,    min=1, max=1e6,      name='Population scale factor',    tip='Multiplier for results (to approximate large populations)')
    sim_pars['n']           = dict(best=5000, min=1, max=max_pop,  name='Population size',            tip='Number of agents simulated in the model')
    sim_pars['n_infected']  = dict(best=10,   min=1, max=max_pop,  name='Initial infections',         tip='Number of initial seed infections in the model')
    sim_pars['n_days']      = dict(best=90,   min=1, max=max_days, name='Number of days to simulate', tip='Number of days to run the simulation for')
    sim_pars['web_int_day'] = dict(best=20,   min=0, max=max_days, name='Intervention start day',     tip='Start day of the intervention (for no intervention, set start day to 0 and effectiveness to 0)')
    sim_pars['web_int_eff'] = dict(best=0.9,  min=0, max=1.0,      name='Intervention effectiveness', tip='Fractional reduction in infectiousness due to intervention')
    sim_pars['seed']        = dict(best=0,    min=0, max=100,      name='Random seed',                tip='Random number seed (set to 0 for different results each time)')

    epi_pars = {}
    epi_pars['beta']          = dict(best=0.015, min=0.0, max=0.2, name='Beta (infectiousness)',         tip ='Probability of infection per contact per day')
    epi_pars['contacts']      = dict(best=20,    min=0.0, max=50,  name='Number of contacts',            tip ='Average number of people each person is in contact with each day')
    epi_pars['web_exp2inf']   = dict(best=4.0,   min=1.0, max=30,  name='Time to infectiousness (days)', tip ='Average number of days between exposure and being infectious')
    epi_pars['web_inf2sym']   = dict(best=1.0,   min=1.0, max=30,  name='Asymptomatic period (days)',    tip ='Average number of days between exposure and developing symptoms')
    epi_pars['web_dur']       = dict(best=10.0,  min=1.0, max=30,  name='Infection duration (days)',     tip ='Average number of days between infection and recovery (viral shedding period)')
    epi_pars['web_timetodie'] = dict(best=22.0,  min=1.0, max=60,  name='Time until death (days)',       tip ='Average number of days between infection and death')
    epi_pars['web_cfr']       = dict(best=0.02,  min=0.0, max=1.0, name='Case fatality rate',            tip ='Proportion of people who become infected who die')


    for parkey,valuedict in regions.items():
        sim_pars[parkey]['best'] = valuedict[region]

    if merge:
        output = {**sim_pars, **epi_pars}
    else:
        output = {'sim_pars': sim_pars, 'epi_pars': epi_pars}

    return output


@app.register_RPC()
def get_version():
    ''' Get the version '''
    output = f'Version {cv.__version__} ({cv.__versiondate__})'
    return output


@app.register_RPC(call_type='upload')
def upload_pars(fname):
    parameters = sc.loadjson(fname)
    if not isinstance(parameters, dict):
        raise TypeError(f'Uploaded file was a {type(parameters)} object rather than a dict')
    if  'sim_pars' not in parameters or 'epi_pars' not in parameters:
        raise KeyError(f'Parameters file must have keys "sim_pars" and "epi_pars", not {parameters.keys()}')
    return parameters


@app.register_RPC()
def run_sim(sim_pars=None, epi_pars=None, show_animation=False, verbose=True):
    ''' Create, run, and plot everything '''

    err = ''

    try:
        # Fix up things that JavaScript mangles
        orig_pars = cv.make_pars()
        defaults = get_defaults(merge=True)
        web_pars = {}
        web_pars['verbose'] = verbose # Control verbosity here


        for key,entry in {**sim_pars, **epi_pars}.items():
            print(key, entry)

            best   = defaults[key]['best']
            minval = defaults[key]['min']
            maxval = defaults[key]['max']

            try:
                web_pars[key] = np.clip(float(entry['best']), minval, maxval)
            except Exception:
                user_key = entry['name']
                user_val = entry['best']
                err1 = f'Could not convert parameter "{user_key}", value "{user_val}"; using default value instead\n'
                print(err1)
                err += err1
                web_pars[key] = best
            if key in sim_pars: sim_pars[key]['best'] = web_pars[key]
            else:               epi_pars[key]['best'] = web_pars[key]

        # Convert durations
        web_pars['dur'] = sc.dcp(orig_pars['dur']) # This is complicated, so just copy it
        web_pars['dur']['exp2inf']['par1']  = web_pars.pop('web_exp2inf')
        web_pars['dur']['inf2sym']['par1']  = web_pars.pop('web_inf2sym')
        web_pars['dur']['crit2die']['par1'] = web_pars.pop('web_timetodie')
        web_dur = web_pars.pop('web_dur')
        for key in ['asym2rec', 'mild2rec', 'sev2rec', 'crit2rec']:
            web_pars['dur'][key]['par1'] = web_dur

        # Add the intervention
        web_pars['interventions'] = []
        if web_pars['web_int_day'] is not None:
            web_pars['interventions'] = cv.change_beta(days=web_pars.pop('web_int_day'), changes=(1-web_pars.pop('web_int_eff')))

        # Handle CFR -- ignore symptoms and set to 1
        prog_pars = cv.get_default_prognoses(by_age=False)
        web_pars['rel_symp_prob']   = 1.0/prog_pars.symp_prob
        web_pars['rel_severe_prob'] = 1.0/prog_pars.severe_prob
        web_pars['rel_crit_prob']   = 1.0/prog_pars.crit_prob
        web_pars['rel_death_prob']  = web_pars.pop('web_cfr')/prog_pars.death_prob

    except Exception as E:
        err2 = f'Parameter conversion failed! {str(E)}\n'
        print(err2)
        err += err2

    # Create the sim and update the parameters
    try:
        sim = cv.Sim()
        sim['prog_by_age'] = False # So the user can override this value
        sim['timelimit'] = max_time # Set the time limit
        if web_pars['seed'] == 0:
            web_pars['seed'] = None # Reset
        sim.update_pars(web_pars)
    except Exception as E:
        err3 = f'Sim creation failed! {str(E)}\n'
        print(err3)
        err += err3

    if verbose:
        print('Input parameters:')
        print(web_pars)

    # Core algorithm
    try:
        sim.run(do_plot=False)
    except Exception as E:
        err4 = f'Sim run failed! {str(E)}\n'
        print(err4)
        err += err4

    if sim.stopped:
        try: # Assume it stopped because of the time, but if not, don't worry
            day = sim.stopped['t']
            time_exceeded = f"The simulation stopped on day {day} because run time limit ({sim['timelimit']} seconds) was exceeded. Please reduce the population size and/or number of days simulated."
            err += time_exceeded
        except:
            pass

    # Core plotting
    graphs = []
    try:
        to_plot = sc.dcp(cv.default_sim_plots)
        for p,title,keylabels in to_plot.enumitems():
            fig = go.Figure()
            for key in keylabels:
                label = sim.results[key].name
                this_color = sim.results[key].color
                y = sim.results[key][:]
                fig.add_trace(go.Scatter(x=sim.results['t'][:], y=y,mode='lines',name=label,line_color=this_color))

            if sim['interventions']:
                interv_day = sim['interventions'][0].days[0]
                if interv_day > 0 and interv_day < sim['n_days']:
                    fig.add_shape(dict(type="line", xref="x", yref="paper", x0=interv_day, x1=interv_day, y0=0, y1=1, name='Intervention', line=dict(width=0.5, dash='dash')))
                    fig.update_layout(annotations=[dict(x=interv_day, y=1.07, xref="x", yref="paper", text="Intervention start", showarrow=False)])

            fig.update_layout(title={'text':title}, xaxis_title='Day', yaxis_title='Count', autosize=True)
            
            output = {'json': fig.to_json(), 'id': str(sc.uuid())}
            d = json.loads(output['json'])
            d['config'] = {'responsive': True}
            output['json'] = json.dumps(d)
            graphs.append(output)

        graphs.append(plot_people(sim))

        if show_animation:
            graphs.append(animate_people(sim))

    except Exception as E:
        err5 = f'Plotting failed! {str(E)}\n'
        print(err5)
        err += err5


    # Create and send output files (base64 encoded content)
    files = {}
    summary = {}
    try:
        datestamp = sc.getdate(dateformat='%Y-%b-%d_%H.%M.%S')


        ss = sim.to_xlsx()
        files['xlsx'] = {
            'filename': f'COVASim_results_{datestamp}.xlsx',
            'content': 'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,' + base64.b64encode(ss.blob).decode("utf-8"),
        }

        json_string = sim.to_json()
        files['json'] = {
            'filename': f'COVASim_results_{datestamp}.txt',
            'content': 'data:application/text;base64,' + base64.b64encode(json_string.encode()).decode("utf-8"),
        }

        # Summary output
        summary = {
            'days': sim.npts-1,
            'cases': round(sim.results['cum_infections'][-1]),
            'deaths': round(sim.results['cum_deaths'][-1]),
        }
    except Exception as E:
        err6 = f'File saving failed! {str(E)}\n'
        print(err6)
        err += err6

    output = {}
    output['err']      = err
    output['sim_pars'] = sim_pars
    output['epi_pars'] = epi_pars
    output['graphs']   = graphs
    output['files']    = files
    output['summary']  = summary

    return output


def get_individual_states(sim, order=True):
    people = sim.people.values()
    if order:
        people = sorted(people, key=lambda x: x.date_exposed if x.date_exposed is not None else np.inf)

    # Order these in order of precedence
    # The last matching quantity will be used
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
         'quantity': 'date_died',
         'color': '#000000',
         'value': 5
         },
    ]

    z = np.zeros((len(people), sim.npts))

    for i, p in enumerate(people):
        for state in states:
            if state['quantity'] is None:
                continue
            elif getattr(p, state['quantity']) is not None:
                z[i, int(getattr(p, state['quantity'])):] = state['value']

    return z, states


def plot_people(sim) -> dict:
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
        interv_day = sim['interventions'][0].days[0]
        if interv_day > 0 and interv_day < sim['n_days']:
            fig.add_shape(dict(type="line", xref="x", yref="paper", x0=interv_day, x1=interv_day, y0=0, y1=1, name='Intervention', line=dict(width=0.5, dash='dash')))
            fig.update_layout(annotations=[dict(x=interv_day, y=1.07, xref="x", yref="paper", text="Intervention start", showarrow=False)])

    fig.update_layout(yaxis_range=(0, sim.n))
    fig.update_layout(title={'text': 'Numbers of people by health state'}, xaxis_title='Day', yaxis_title='People', autosize=True)

    output = {'json': fig.to_json(), 'id': str(sc.uuid())}
    d = json.loads(output['json'])
    d['config'] = {'responsive': True}
    output['json'] = json.dumps(d)

    return output


def animate_people(sim) -> dict:
    z, states = get_individual_states(sim, order=False)

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
            "prefix": "Day:",
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

    fig.update_layout(
        plot_bgcolor='#fff'
    )

    fig.update_layout(title={'text': 'Epidemic over time'})

    output = {'json': fig.to_json(), 'id': str(sc.uuid())}
    d = json.loads(output['json'])
    d['config'] = {'responsive': True}
    output['json'] = json.dumps(d)

    return output

#%% Run the server using Flask
if __name__ == "__main__":

    os.chdir(sc.thisdir(__file__))

    if len(sys.argv) > 1:
        app.config['SERVER_PORT'] = int(sys.argv[1])
    else:
        app.config['SERVER_PORT'] = 8188
    if len(sys.argv) > 2:
        autoreload = int(sys.argv[2])
    else:
        autoreload = 1

    app.run(autoreload=autoreload)
