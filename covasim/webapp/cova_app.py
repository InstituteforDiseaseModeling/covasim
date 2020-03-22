'''
Sciris app to run the web interface.
'''

# Key imports
import os
import sys
import pylab as pl
import plotly.graph_objects as go
import sciris as sc
import covasim as cv
import base64 # Download/upload-specific import


# Check requirements, and if met, import scirisweb
cv._requirements.check_scirisweb(die=True)
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
    sim_pars['web_int_day'] = dict(best=20,   min=0, max=max_days, name='Intervention start day',     tip='Start day of the intervention (can be blank)')
    sim_pars['web_int_eff'] = dict(best=0.9,  min=0, max=1.0,      name='Intervention effectiveness', tip='Reduction in infectiousness due to intervention')
    sim_pars['seed']        = dict(best=0,    min=0, max=100,      name='Random seed',                tip='Random number seed (set to 0 for different results each time)')

    epi_pars = {}
    epi_pars['beta']        = dict(best=0.015, min=0.0, max=0.2, name='Beta (infectiousness)',     tip='Probability of infection per contact per day')
    epi_pars['contacts']    = dict(best=20,    min=0.0, max=50,  name='Number of contacts',        tip='Average number of people each person is in contact with each day')
    epi_pars['serial']      = dict(best=4.0,   min=1.0, max=30,  name='Serial interval (days)',    tip='Average number of days between exposure and being infectious')
    epi_pars['incub']       = dict(best=5.0,   min=1.0, max=30,  name='Incubation period (days)',  tip='Average number of days between exposure and developing symptoms')
    epi_pars['dur']         = dict(best=8.0,   min=1.0, max=30,  name='Infection duration (days)', tip='Average number of days between infection and recovery (viral shedding period)')
    epi_pars['timetodie']   = dict(best=22.0,  min=1.0, max=60,  name='Time until death (days)',   tip='Average number of days between infection and death')
    epi_pars['default_cfr'] = dict(best=0.02,  min=0.0, max=1.0, name='Case fatality rate',        tip='Proportion of people who become infected who die')


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
def run_sim(sim_pars=None, epi_pars=None, verbose=True):
    ''' Create, run, and plot everything '''

    err = ''

    try:
        # Fix up things that JavaScript mangles
        defaults = get_defaults(merge=True)
        web_pars = {}
        web_pars['verbose'] = verbose # Control verbosity here

        for key,entry in {**sim_pars, **epi_pars}.items():
            print(key, entry)

            best   = defaults[key]['best']
            minval = defaults[key]['min']
            maxval = defaults[key]['max']

            try:
                web_pars[key] = pl.median([float(entry['best']), minval, maxval])
            except Exception:
                user_key = entry['name']
                user_val = entry['best']
                err1 = f'Could not convert parameter "{user_key}", value "{user_val}"; using default value instead\n'
                print(err1)
                err += err1
                web_pars[key] = best
            if key in sim_pars: sim_pars[key]['best'] = web_pars[key]
            else:               epi_pars[key]['best'] = web_pars[key]

        # Handle special cases
        web_pars['interv_days'] = [web_pars.pop('web_int_day')] # Turn from scalar to list
        web_pars['interv_effs'] = [1-web_pars.pop('web_int_eff')] # Turn from scalar to list and take inverse

    except Exception as E:
        err2 = f'Parameter conversion failed! {str(E)}\n'
        print(err2)
        err += err2

    # Create the sim and update the parameters
    try:
        sim = cv.Sim()
        sim['cfr_by_age'] = False # So the user can override this value
        sim['timelimit'] = max_time # Set the time limit
        sim.update_pars(pars=web_pars)
        if web_pars['seed'] != 0:
            sim.set_seed(int(web_pars['seed']))
        else:
            sim.set_seed(None)
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

        to_plot = sc.dcp(cv.to_plot)
        for p,title,keylabels in to_plot.enumitems():
            fig = go.Figure()
            colors = sc.gridcolors(len(keylabels))
            for i,key,label in keylabels.enumitems():
                this_color = 'rgb(%d,%d,%d)' % (255*colors[i][0],255*colors[i][1],255*colors[i][2])
                y = sim.results[key][:]
                fig.add_trace(go.Scatter(x=sim.results['t'][:], y=y,mode='lines',name=label,line_color=this_color))

            if len(sim['interv_days']):
                interv_day = sim['interv_days'][0]
                if interv_day > 0 and interv_day < sim['n_days']:
                    fig.add_shape(dict(type="line", xref="x", yref="paper", x0=interv_day, x1=interv_day, y0=0, y1=1, name='Intervention', line=dict(width=0.5, dash='dash')))
            fig.update_layout(title={'text':title}, xaxis_title='Day', yaxis_title='Count', autosize=True)
            graphs.append({'json':fig.to_json(),'id':str(sc.uuid())})
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

        json = sim.to_json()
        files['json'] = {
            'filename': f'COVASim_results_{datestamp}.txt',
            'content': 'data:application/text;base64,' + base64.b64encode(json.encode()).decode("utf-8"),
        }

        # Summary output
        summary = {
            'days': sim.npts-1,
            'cases': round(sim.results['cum_exposed'][-1]),
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

    app.run(autoreload=True)
