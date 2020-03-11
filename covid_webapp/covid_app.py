'''
Sciris app to run the web interface.
'''

# Imports
import os
import sys
import pylab as pl
import mpld3
import sciris as sc
import scirisweb as sw
import voi_sim as vs

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
    
    base = {
        'n_children':        {'name':'Number of children',                             'best':1000},
        'max_time':          {'name':'Simulation duration (months)',                   'best':180},
        'RI_coverage':       {'name':'Routine immunization coverage',                  'best':0.4},
        'reactive_campaigns':{'name':'Use reactive campaigns? (0=false, 1=true)',      'best':1},
        'p_vaccinate_0':     {'name':'Probability to vaccinate in campaign, 1st dose', 'best':0.4},
        'p_vaccinate_1':     {'name':'Probability to vaccinate in campaign, 2nd dose', 'best':0.4},
    }
    
    simple = {'p_track':            {'name': 'Proportion of children who accept the microarray tracker',     'best': 0.80, 'min': 0.0, 'max': 1.0},
              'cost_vaccine':       {'name': 'Cost per vaccine dose per child (US$)',                        'best': 0.80, 'min': 0.0, 'max': 1.0},
              'cost_tracker_apply': {'name': 'Cost to apply the microarray tracker to a single child (US$)', 'best': 0.50, 'min': 0.0, 'max': 1.0},
              'cost_tracker_read':  {'name': 'Cost to read the microarray tracker for a single child (US$)', 'best': 0.10, 'min': 0.0, 'max': 1.0},
            }
    
    advanced = {
      'microarray':
             {'p_track':            {'name': 'Tracker delivery TPR',             'best': 0.80, 'min': 0.0, 'max': 1.0, 'tooltip':'Probability of receiving tracker with vaccination'},
              'p_track_f':          {'name': 'Tracker delivery FPR',             'best': 0.01, 'min': 0.0, 'max': 1.0, 'tooltip':'Probability of receiving tracker without vaccination'},
              'p_report':           {'name': 'Tracker reading TPR',              'best': 0.99, 'min': 0.0, 'max': 1.0, 'tooltip':'Probability of someone with tracker being correctly identified'},
              'p_report_f':         {'name': 'Tracker reading FPR',              'best': 0.01, 'min': 0.0, 'max': 1.0, 'tooltip':'Probability of someone without tracker being incorrectly identified'},
              'tracker_fading':     {'name': 'Tracker fading (per month)',       'best': 0.05, 'min': 0.0, 'max': 1.0, 'tooltip':'Rate at which tracker becomes invisible (probability per month)'},
              'cost_vaccine':       {'name': 'Cost per vaccine dose (US$)',      'best': 0.80, 'min': 0.0, 'max': 1.0, 'tooltip':'Cost to deliver a single dose of vaccine to a child'},
              'cost_tracker_apply': {'name': 'Cost to apply each tracker (US$)', 'best': 0.50, 'min': 0.0, 'max': 1.0, 'tooltip':'Cost to deliver a single tracker to a child'},
              'cost_tracker_read':  {'name': 'Cost to read tracker (US$/child)', 'best': 0.10, 'min': 0.0, 'max': 1.0, 'tooltip':'Cost to check whether or not a child has a tracker'},
             },
    'fingermark':
             {'p_track':            {'name': 'Tracker delivery TPR',             'best': 0.90},
              'p_track_f':          {'name': 'Tracker delivery FPR',             'best': 0.10},
              'p_report':           {'name': 'Tracker reading TPR',              'best': 0.80},
              'p_report_f':         {'name': 'Tracker reading FPR',              'best': 0.10},
              'tracker_fading':     {'name': 'Tracker fading (per month)',       'best': 0.20},
              'cost_vaccine':       {'name': 'Cost per vaccine dose (US$)',      'best': 0.80},
              'cost_tracker_apply': {'name': 'Cost to apply each tracker (US$)', 'best': 0.01},
              'cost_tracker_read':  {'name': 'Cost to read tracker (US$/child)', 'best': 0.01},
             },
    }

    # Populate missing fields
    for key1 in advanced['microarray'].keys():
      for key2 in advanced['microarray'][key1].keys():
        if key2 != 'best':
          advanced['fingermark'][key1][key2] = advanced['microarray'][key1][key2]
          
    sweep = {
            'pars': {
                     'p_track':           {'name':'Proportion of children who accept the microarray tracker', 'min': 0.0, 'max': 1.0},
                     'RI_coverage':       {'name':'Vaccine coverage due to routine immunization',             'min': 0.0, 'max': 1.0},
                     'p_vaccinate_0':     {'name':'Probability to vaccinate in campaign, 1st dose',           'min': 0.0, 'max': 1.0},
                     'p_vaccinate_1':     {'name':'Probability to vaccinate in campaign, 2nd dose',           'min': 0.0, 'max': 1.0},
                     'cost_vaccine':      {'name':'Cost per vaccine dose (US$)',                              'min': 0.0, 'max': 10.0},
                },
            'meta': {
                     'par': 'p_track',
                     'reps': 4,
                     'steps': 5,
                    }
            }
    
    output = {'base': base, 'simple': simple, 'advanced': advanced, 'sweep':sweep}
    return output


def get_no_vacc():
    ''' Get the no-vaccination scenario '''
    output = {
                    'name': 'No vaccination',
                    'p_track': 0.0, 
                    'p_track_f': 0.0,
                    'p_report': 0.0,
                    'p_report_f': 0.0,
                    'tracker_fading': 0.0,
                    'cost_tracker_apply': 0.0,
                    'cost_tracker_read': 0.0,
                    'p_vaccinate_0': 0,
                    'p_vaccinate_1': 0,
                    'p_vaccinate_2plus': 0, # Vaccination does not depend on finger marks
                    'RI_coverage': 0,
                    'track_at_RI': 0
                }
    return output


@app.register_RPC()
def get_version():
    ''' Get the version '''
    output = f'{vs.__version__} ({vs.__versiondate__})'
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
def plot_sim(session_id, base_pars, pars, verbose=True):
    ''' Create, run, and plot everything '''
    
    # Fix up things that JavaScript mangles
    session_id = str(session_id)
    base_pars = sc.odict(base_pars)
    pars = sc.odict(pars)
    
    sim_pars = sc.odict()
    sim_pars['verbose'] = verbose # Control verbosity here
    for key,entry in base_pars.items() + pars.items():
        sim_pars[key] = float(entry['best'])
    
    # Handle sessions
    try:
        sim = app.sessions[session_id]['sim']
        sim.update_pars(pars=sim_pars)
        print(f'Loaded sim session {session_id}')
    except Exception as E:
        sim = vs.Sim(pars=sim_pars)
        app.sessions[session_id].sim = sim
        print(f'Added sim session {session_id} ({str(E)})')
    
    if verbose:
        print('Input parameters:')
        print(sim_pars)
     
    # Core algorithm
    sim.run()
    fig = sim.plot(figargs={'figsize':(8,8)}) # Plot the sim
    mpld3.plugins.connect(fig, mpld3.plugins.MousePosition(fontsize=12, fmt='.4r')) # Add data cursor
    
    # Convert graph to FE
    graphjson = sw.mpld3ify(fig, jsonify=False)  # Convert to dict
    return graphjson  # Return the JSON representation of the Matplotlib figure


@app.register_RPC()
def plot_multisim(session_id, base_pars, pars, verbose=False, include_no_vacc=False):
    ''' Create, run, and plot everything '''
    
    # Fix up things that JavaScript mangles
    session_id = str(session_id)
    base_pars = sc.odict(base_pars)
    pars = sc.odict(pars)
    
    key_list = ['fingermark', 'microarray'] # Hard-code here since the structure returned from the FE can't be trusted
    
    parsets = sc.odict()
    if include_no_vacc:
        parsets['no_vacc'] = get_no_vacc()
    sim_names = {'no_vacc':'No vaccination', 'fingermark':'Finger mark', 'microarray':'Microarray'} # Defined here so other parameters can be iterated over
    for key1 in key_list:
        parset = sc.odict(pars[key1])
        parsets[key1] = {'name':sim_names[key1]}
        parsets[key1]['verbose'] = verbose # Control verbosity here
        for key2,entry in base_pars.items() + parset.items():
            parsets[key1][key2] = float(entry['best'])
            if include_no_vacc and key2 not in parsets['no_vacc'].keys():
                parsets['no_vacc'][key2] = float(entry['best'])
    
    # Handle sessions
    try:
        multisim = app.sessions[session_id]['multisim']
        multisim.update_parsets(parsets=parsets)
        print(f'Loaded multisim session {session_id}')
    except Exception:
        multisim = vs.Multisim(parsets=parsets)
        app.sessions[session_id].multisim = multisim
        print(f'Added multisim session {session_id}')
     
    # Core algorithm
    multisim.run()
    fig = multisim.plot(figargs={'figsize':(8,14)}) # Plot the sim
    mpld3.plugins.connect(fig, mpld3.plugins.MousePosition(fontsize=12, fmt='.4r')) # Add data cursor
    
    # Convert graph to FE
    graphjson = sw.mpld3ify(fig, jsonify=False)  # Convert to dict
    return graphjson  # Return the JSON representation of the Matplotlib figure


@app.register_RPC()
def plot_sweep(session_id, base_pars, pars, sweep_pars, verbose=False, include_no_vacc=True):
    ''' Create, run, and plot everything '''
    
    # Fix up things that JavaScript mangles
    session_id = str(session_id)
    base_pars = sc.odict(base_pars)
    pars = sc.odict(pars)
    
    key_list = ['fingermark', 'microarray'] # Hard-code here since the structure returned from the FE can't be trusted
    
    parsets = sc.odict()
    if include_no_vacc:
        parsets['no_vacc'] = get_no_vacc()
    sim_names = {'no_vacc':'No vaccination', 'fingermark':'Finger mark', 'microarray':'Microarray'} # Defined here so other parameters can be iterated over
    for key1 in key_list:
        parset = sc.odict(pars[key1])
        parsets[key1] = {'name':sim_names[key1]}
        parsets[key1]['verbose'] = verbose # Control verbosity here
        if include_no_vacc:
            parsets['no_vacc']['verbose'] = verbose # Control verbosity here
        for key2,entry in base_pars.items() + parset.items():
            parsets[key1][key2] = float(entry['best'])
            if include_no_vacc and key2 not in parsets['no_vacc'].keys():
                parsets['no_vacc'][key2] = float(entry['best'])
    
    # Handle sessions
    try:
        sweep = app.sessions[session_id]['sweep']
        sweep.update_parsets(parsets=parsets)
        print(f'Loaded sweep session {session_id}')
    except Exception:
        par    = sweep_pars['meta']['par']
        minval = float(sweep_pars['pars'][par]['min'])
        maxval = float(sweep_pars['pars'][par]['max'])
        steps  = int(sweep_pars['meta']['steps'])
        reps   = int(sweep_pars['meta']['reps'])
        label  = sweep_pars['pars'][par]['name']
        sweep = vs.Sweep(parsets=parsets, sweep={par:pl.linspace(minval, maxval, steps)}, reps=reps, xlabel=label)
        app.sessions[session_id].sweep = sweep
        print(f'Added sweep session {session_id}')
     
    # Core algorithm
    sweep.run()
    fig = sweep.plot(figargs={'figsize':(8,14)}) # Plot the sim
    mpld3.plugins.connect(fig, mpld3.plugins.MousePosition(fontsize=12, fmt='.4r')) # Add data cursor
    
    # Convert graph to FE
    graphjson = sw.mpld3ify(fig, jsonify=False)  # Convert to dict
    return graphjson  # Return the JSON representation of the Matplotlib figure



#%% Run the server
if __name__ == "__main__":
    app.run(autoreload=True)