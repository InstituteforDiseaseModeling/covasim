'''
Sciris app to run the web interface.
'''

# Imports
import os
import sys
import sciris as sc
import scirisweb as sw
import plotly.graph_objects as go
import pylab as pl
import covasim.cova_webapp as cw # Short for "Covid webapp model"

# Change to current folder and create the app
app = sw.ScirisApp(__name__, name="COVASim")
app.sessions = dict() # For storing user data
flask_app = app.flask_app

#%% Define the API

@app.register_RPC()
def get_defaults(region=None, merge=False):
    ''' Get parameter defaults '''

    if region is None:
        region = 'Example'

    max_pop = 1e5
    max_days = 365

    regions = {
        'scale': {
            'Example': 1,
            'Seattle': 25,
            # 'Wuhan': 200,
        },
        'n': {
            'Example': 5000,
            'Seattle': 10000,
            # 'Wuhan': 1,
        },
        'n_days': {
            'Example': 90,
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
    sim_pars['scale']            = {'best':1,    'min':1,   'max':1e9,      'name':'Population scale factor'}
    sim_pars['n']                = {'best':5000, 'min':1,   'max':max_pop,  'name':'Population size'}
    sim_pars['n_infected']       = {'best':10,   'min':1,   'max':max_pop,  'name':'Initial infections'}
    sim_pars['n_days']           = {'best':90,   'min':1,   'max':max_days, 'name':'Duration (days)'}
    sim_pars['intervene']        = {'best':20,   'min':-1,  'max':max_days, 'name':'Intervention start (day)'}
    sim_pars['unintervene']      = {'best':40,   'min':-1,  'max':max_days, 'name':'Intervention end (day)'}
    sim_pars['intervention_eff'] = {'best':0.9,  'min':0.0, 'max':1.0,      'name':'Intervention effectiveness'}
    sim_pars['seed']             = {'best':1,    'min':1,   'max':1e9,      'name':'Random seed'}

    epi_pars = {}
    epi_pars['r0']        = {'best':2.05,  'min':0.0, 'max':20.0, 'name':'R0 (infectiousness)'}
    epi_pars['contacts']  = {'best':20,   'min':0.0, 'max':100,  'name':'Number of contacts'}
    epi_pars['incub']     = {'best':4.5,  'min':1.0, 'max':30,   'name':'Incubation period (days)'}
    epi_pars['incub_std'] = {'best':1.0,  'min':0.0, 'max':30,   'name':'Incubation variability (days)'}
    epi_pars['dur']       = {'best':8.0,  'min':1.0, 'max':30,   'name':'Infection duration (days)'}
    epi_pars['dur_std']   = {'best':2.0,  'min':0.0, 'max':30,   'name':'Infection variability (days)'}
    epi_pars['cfr']       = {'best':0.02, 'min':0.0, 'max':1.0,  'name':'Case fatality rate'}
    epi_pars['timetodie'] = {'best':22.0, 'min':1.0, 'max':60,   'name':'Days until death'}

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
    output = f'{cw.__version__} ({cw.__versiondate__})'
    return output


@app.register_RPC()
def get_sessions(session_id=None):
    ''' Get the sessions '''
    try:
        session_list = app.sessions.keys()
        if not session_id:
            session_id = len(session_list)+1
            session_list.append(session_id)
            app.sessions[str(session_id)] = sc.objdict()
            print(f'Created session {session_id}')
        output = {'session_id':session_id, 'session_list':session_list, 'err':''}
    except Exception as E:
        err = f'Session retrieval failed! ({str(E)})'
        print(err)
        output = {'session_id':1, 'session_list':[1], 'err':err}
    return output


@app.register_RPC()
def plot_sim(sim_pars=None, epi_pars=None, verbose=True):
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
            userval = float(entry['best'])
            minval = defaults[key]['min']
            maxval = defaults[key]['max']
            best = pl.median([userval, minval, maxval])
            pars[key] = best
            if key in sim_pars: sim_pars[key]['best'] = best
            else:               epi_pars[key]['best'] = best
    except Exception as E:
        err1 = f'Parameter conversion failed! {str(E)}'
        print(err1)
        err += err1

    # Handle sessions
    sim = cw.Sim()
    sim.update_pars(pars=pars)

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

    if sim.data is not None:
        data_mapping = {
            'cum_diagnosed': pl.cumsum(sim.data['new_positives']),
            'tests':         sim.data['new_tests'],
            'diagnoses':     sim.data['new_positives'],
            }
    else:
        data_mapping = {}

    output = {}
    output['err'] = err
    output['sim_pars'] = sim_pars
    output['epi_pars'] = epi_pars
    output['graphs'] = []

    # If prevalence stays low, don't plot susceptibles
    to_plot = sc.dcp(cw.to_plot)
    max_exposed = sim.results['cum_exposed'].max()
    threshold = prev_threshold*sim.results['n_susceptible'].max()
    if max_exposed < threshold:
        to_plot['Total counts'].pop('n_susceptible')

    for p,title,keylabels in to_plot.enumitems():
        fig = go.Figure()
        colors = sc.gridcolors(len(keylabels))
        for i,key,label in keylabels.enumitems():
            this_color = 'rgb(%d,%d,%d)' % (255*colors[i][0],255*colors[i][1],255*colors[i][2])
            y = sim.results[key]
            fig.add_trace(go.Scatter(x=sim.results['t'], y=y,mode='lines',name=label,line_color=this_color))
            if key in data_mapping:
                fig.add_trace(go.Scatter(x=sim.data['day'], y=data_mapping[key],mode='markers',name=label,fill_color=this_color))
        fig.update_layout(title={'text':title}, #, 'xanchor':'center', 'x':0.5},
                          xaxis_title='Day',
                          yaxis_title='Count',
                            autosize = True,
        )
        output['graphs'].append({'json':fig.to_json(),'id':str(sc.uuid())})

    return output


#%% Run the server
if __name__ == "__main__":

    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    if len(sys.argv) > 1:
        app.config['SERVER_PORT'] = int(sys.argv[1])
    else:
        app.config['SERVER_PORT'] = 8188
    if len(sys.argv) > 2:
        autoreload = int(sys.argv[2])
    else:
        autoreload = 1

    app.run(autoreload=True)