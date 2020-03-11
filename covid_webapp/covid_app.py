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

    pars = cs.make_pars()

    output = {'pars': pars}
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
def plot_sim(session_id, base_pars=None, pars=None, verbose=True):
    ''' Create, run, and plot everything '''

    # Fix up things that JavaScript mangles
    session_id = str(session_id)
    # base_pars = sc.odict(base_pars)
    # pars = sc.odict(pars)

    sim_pars = sc.odict()
    # sim_pars['verbose'] = verbose # Control verbosity here
    # for key,entry in base_pars.items() + pars.items():
    #     sim_pars[key] = float(entry['best'])

    # Handle sessions
    try:
        sim = app.sessions[session_id]['sim']
        # sim.update_pars(pars=sim_pars)
        print(f'Loaded sim session {session_id}')
    except Exception as E:
        sim = cs.Sim(pars=sim_pars)
        app.sessions[session_id].sim = sim
        print(f'Added sim session {session_id} ({str(E)})')

    if verbose:
        print('Input parameters:')
        print(sim_pars)

    # Core algorithm
    sim.run(do_plot=False)
    fig = sim.plot(fig_args={'figsize':(8,8)}) # Plot the sim
    mpld3.plugins.connect(fig, mpld3.plugins.MousePosition(fontsize=12, fmt='.4r')) # Add data cursor

    # Convert graph to FE
    graphjson = sw.mpld3ify(fig, jsonify=False)  # Convert to dict
    return graphjson  # Return the JSON representation of the Matplotlib figure



#%% Run the server
if __name__ == "__main__":
    app.run(autoreload=True)