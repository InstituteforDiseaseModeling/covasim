'''
Sciris app to run the web interface.
'''

# Imports
import os
import io
import sys
import base64
import zipfile
import sciris as sc
import scirisweb as sw
import plotly.graph_objects as go
import pylab as pl
import covasim.base as cw # Short for "Covid webapp model"
import json

# Create the app
app = sw.ScirisApp(__name__, name="Covasim")
app.sessions = dict() # For storing user data
flask_app = app.flask_app

#%% Define the API

@app.register_RPC()
def get_defaults(region=None, merge=False):
    ''' Get parameter defaults '''

    if region is None:
        region = 'Example'

    max_pop = 10e3
    max_days = 90

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
            'Example': 10,
            'Seattle': 4,
            # 'Wuhan': 10,
        },
        'intervene': {
            'Example': 20,
            'Seattle': 17,
            # 'Wuhan': 1,
        },
        'unintervene': {
            'Example': 40,
            'Seattle': -1,
            # 'Wuhan': 1,
        },
        'intervention_eff': {
            'Example': 0.5,
            'Seattle': 0.0,
            # 'Wuhan': 0.9,
        },
    }

    sim_pars = {}
    sim_pars['scale']            = dict(best=1,    min=1,   max=1e9,      name='Population scale factor',    tip='Multiplier for results (to approximate large populations)')
    sim_pars['n']                = dict(best=5000, min=1,   max=max_pop,  name='Population size',            tip='Number of agents simulated in the model')
    sim_pars['n_infected']       = dict(best=10,   min=1,   max=max_pop,  name='Initial infections',         tip='Number of initial seed infections in the model')
    sim_pars['n_days']           = dict(best=90,   min=1,   max=max_days, name='Duration (days)',            tip='Number of days to run the simulation for')
    sim_pars['intervene']        = dict(best=20,   min=-1,  max=max_days, name='Intervention start (day)',   tip='Start day of the intervention (can be blank)')
    sim_pars['unintervene']      = dict(best=40,   min=-1,  max=max_days, name='Intervention end (day)',     tip='Final day of intervention (can be blank)')
    sim_pars['intervention_eff'] = dict(best=0.9,  min=0.0, max=1.0,      name='Intervention effectiveness', tip='Change in infection rate due to intervention')
    sim_pars['seed']             = dict(best=1,    min=1,   max=1e9,      name='Random seed',                tip='Random number seed (leave blank for random results)')

    epi_pars = {}
    epi_pars['beta']      = dict(best=0.015, min=0.0, max=0.1,  name='Beta (infectiousness)',         tip='Probability of infection per contact per day')
    epi_pars['contacts']  = dict(best=20,    min=0.0, max=100,  name='Number of contacts',            tip='Number of people, on average, each person is in contact with')
    epi_pars['incub']     = dict(best=4.5,   min=1.0, max=30,   name='Incubation period (days)',      tip='Average length of time of incubation before symptoms')
    epi_pars['incub_std'] = dict(best=1.0,   min=0.0, max=30,   name='Incubation variability (days)', tip='Standard deviation of incubation period')
    epi_pars['dur']       = dict(best=8.0,   min=1.0, max=30,   name='Infection duration (days)',     tip='Average length of time of infection (viral shedding)')
    epi_pars['dur_std']   = dict(best=2.0,   min=0.0, max=30,   name='Infection variability (days)',  tip='Standard deviation of infection period')
    epi_pars['cfr']       = dict(best=0.02,  min=0.0, max=1.0,  name='Case fatality rate',            tip='Proportion of people who become infected who die')
    epi_pars['timetodie'] = dict(best=22.0,  min=1.0, max=60,   name='Days until death',              tip='Average length of time between infection and death')

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
    output = f'Version {cw.__version__} ({cw.__versiondate__})'
    return output


@app.register_RPC(call_type='download')
def download_pars(sim_pars, epi_pars):
    datestamp = sc.getdate(dateformat='%Y-%b-%d_%H.%M.%S')
    filename = f'COVASim_parameters_{datestamp}.json'
    d = {'sim_pars':sim_pars,'epi_pars':epi_pars}
    s = json.dumps(d,indent=2)
    output = (io.BytesIO(("'%s'" % (s)).encode()), filename)
    return output


@app.register_RPC(call_type='upload')
def upload_pars(fname):
    with open(fname,'r') as f:
        s = f.read()
    parameters = json.loads(s[1:-1])
    if not isinstance(parameters, dict):
        raise TypeError(f'Uploaded file was a {type(parameters)} object rather than a dict')
    if  'sim_pars' not in parameters or 'epi_pars' not in parameters:
        raise KeyError(f'Parameters file must have keys "sim_pars" and "epi_pars", not {parameters.keys()}')
    return parameters


@app.register_RPC()
def run_sim(sim_pars=None, epi_pars=None, verbose=True):
    ''' Create, run, and plot everything '''

    prev_threshold = 0.20 # Don't plot susceptibles if prevalence never gets above this threshold

    err = ''

    try:
        # Fix up things that JavaScript mangles
        defaults = get_defaults(merge=True)
        pars = {}
        pars['verbose'] = verbose # Control verbosity here
        for key,entry in {**sim_pars, **epi_pars}.items():
            print(key, entry)
            minval = defaults[key]['min']
            maxval = defaults[key]['max']
            if entry['best']:
                pars[key] = pl.median([float(entry['best']), minval, maxval])
            else:
                pars[key] = None
            if key in sim_pars: sim_pars[key]['best'] = pars[key]
            else:               epi_pars[key]['best'] = pars[key]
    except Exception as E:
        err1 = f'Parameter conversion failed! {str(E)}'
        print(err1)
        err += err1

    # Handle sessions
    sim = cw.Sim()
    sim.update_pars(pars=pars)
    if pars['seed'] is not None:
        sim.set_seed(int(pars['seed']))
    else:
        sim.set_seed()

    if verbose:
        print('Input parameters:')
        print(pars)

    # Core algorithm
    try:
        sim.run(do_plot=False)
    except Exception as E:
        err3 = f'Sim run failed! ({str(E)})'
        print(err3)
        err += err3

    output = {}
    output['err'] = err
    output['sim_pars'] = sim_pars
    output['epi_pars'] = epi_pars
    output['graphs'] = []

    # Core plotting
    to_plot = sc.dcp(cw.to_plot)
    for p,title,keylabels in to_plot.enumitems():
        fig = go.Figure()
        colors = sc.gridcolors(len(keylabels))
        for i,key,label in keylabels.enumitems():
            this_color = 'rgb(%d,%d,%d)' % (255*colors[i][0],255*colors[i][1],255*colors[i][2])
            y = sim.results[key][:]
            fig.add_trace(go.Scatter(x=sim.results['t'][:], y=y,mode='lines',name=label,line_color=this_color))
        fig.update_layout(title={'text':title}, xaxis_title='Day', yaxis_title='Count', autosize=True)
        output['graphs'].append({'json':fig.to_json(),'id':str(sc.uuid())})

    # Create and send output files
    # base64 encoded content
    datestamp = sc.getdate(dateformat='%Y-%b-%d_%H.%M.%S')
    output['files'] = {}

    ss = sim.export_xlsx()
    output['files']['xlsx'] = {
        'filename': f'COVASim_results_{datestamp}.xlsx',
        'content': base64.b64encode(ss.blob).decode("utf-8"),
    }

    result_json = sim.export_json()
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as f:
        f.writestr('results.txt', result_json)
    zip_buffer.flush()
    zip_buffer.seek(0)

    output['files']['json'] = {
        'filename': f'COVASim_results_{datestamp}.zip',
        'content': base64.b64encode(zip_buffer.read()).decode("utf-8"),
    }

    # Summary output
    output['summary'] = {
        'days': sim.npts-1,
        'cases': round(sim.results['cum_exposed'][-1]),
        'deaths': round(sim.results['cum_deaths'][-1]),
    }

    return output


#%% Run the server
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