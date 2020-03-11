'''
Sciris app to run the web interface.
'''

# Imports
import os
import sys
import mpld3
import sciris as sc
import scirisweb as sw
import covid_seattle as cs

# Change to current folder and create the app
os.chdir(os.path.abspath(os.path.dirname(__file__)))
if len(sys.argv)>1:  port = int(sys.argv[1])
else:                port = 8188
if len(sys.argv)>2:  autoreload = int(sys.argv[2])
else:                autoreload = 1
app = sw.ScirisApp(__name__, name="Covid-ABM", server_port=port)
app.sessions = sc.odict() # For storing user data


#%% Define the API

@app.register_RPC()
def get_defaults():
    ''' Get parameter defaults '''

    sim_pars = {}
    sim_pars['scale']          = {'best':1,    'name':'Scaling factor'}
    sim_pars['n']              = {'best':10000,'name':'Population size'}
    sim_pars['n_infected']     = {'best':10,   'name':'Seed infections'}
    sim_pars['n_days']         = {'best':45,   'name':'Duration (days)'}
    sim_pars['seed']           = {'best':1,    'name':'Random seed'}
    sim_pars['intervene']      = {'best':-1,   'name':'Intervention start'}
    sim_pars['unintervene']    = {'best':-1,   'name':'Intervention end'}
    sim_pars['intervention_eff'] = {'best':0.0, 'name':'Intervention effectiveness'}

    epi_pars = {}
    epi_pars['r_contact']      = {'best':0.025,'name':'Infectiousness (beta)'}
    epi_pars['contacts']       = {'best':10,   'name':'Number of contacts'}
    epi_pars['incub']          = {'best':4.0,  'name':'Incubation period'}
    epi_pars['dur']            = {'best':8.0,  'name':'Infection duration'}
    epi_pars['cfr']            = {'best':0.02, 'name':'Case fatality rate'}
    epi_pars['timetodie']      = {'best':22.0, 'name':'Days until death'}

    output = {'sim_pars': sim_pars, 'epi_pars': epi_pars}
    return output


@app.register_RPC()
def get_version():
    ''' Get the version '''
    output = f'{cs.__version__} ({cs.__versiondate__})'
    return output


@app.register_RPC()
def get_sessions(session_id=None):
    ''' Get the sessions '''
    session_list = app.sessions.keys()
    if not session_id:
        session_id = len(session_list)+1
        session_list.append(session_id)
        app.sessions[str(session_id)] = sc.objdict()
        print(f'Created session {session_id}')
    output = {'session_id':session_id, 'session_list':session_list}
    return output


@app.register_RPC()
def plot_sim(session_id, sim_pars=None, epi_pars=None, verbose=True):
    ''' Create, run, and plot everything '''

    # Fix up things that JavaScript mangles
    session_id = str(session_id)

    sim_pars = sc.odict(sim_pars)
    epi_pars = sc.odict(epi_pars)
    pars = {}
    pars['verbose'] = verbose # Control verbosity here
    for key,entry in sim_pars.items() + epi_pars.items():
        pars[key] = float(entry['best'])

    # Handle sessions
    try:
        sim = app.sessions[session_id]['sim']
        sim.update_pars(pars=pars)
        print(f'Loaded sim session {session_id}')
    except Exception as E:
        sim = cs.Sim()
        sim.update_pars(pars=pars)
        app.sessions[session_id].sim = sim
        print(f'Added sim session {session_id} ({str(E)})')

    if verbose:
        print('Input parameters:')
        print(pars)

    # Core algorithm
    sim.run(do_plot=False)

    # Plotting
    fig_args = {'figsize':(8,8)}
    axis_args = {'left':0.15, 'bottom':0.1, 'right':0.9, 'top':0.95, 'wspace':0.2, 'hspace':0.25}
    kwargs = dict(fig_args=fig_args, axis_args=axis_args, font_size=12, use_grid=False)
    fig = sim.plot(**kwargs) # Plot the sim
    mpld3.plugins.connect(fig, mpld3.plugins.MousePosition(fontsize=12, fmt='.4r')) # Add data cursor

    # Convert graph to FE
    graphjson = sw.mpld3ify(fig, jsonify=False)  # Convert to dict
    return graphjson  # Return the JSON representation of the Matplotlib figure



#%% Run the server
if __name__ == "__main__":
    app.run(autoreload=True)