'''
Additional analysis functions that are not part of the core Covasim workflow,
but which are useful for particular investigations. Currently, this just consists
of the transmission tree.
'''

import pylab as pl
import numpy as np
import pandas as pd
import sciris as sc


__all__ = ['TransTree']


def import_nx():
    ''' Fail gracefully if NetworkX is called for but is not found '''
    try:
        import networkx as nx # Optional import
    except ImportError as E:
        errormsg = f'WARNING: Could not import networkx ({str(E)}). Some functionality will be disabled. If desired, please install manually, e.g. "pip install networkx".'
        print(errormsg)
        nx = None
    return nx


class TransTree(sc.prettyobj):
    '''
    A class for holding a transmission tree. There are several different representations
    of the transmission tree: "infection_log" is copied from the people object and is the
    simplest representation. "detailed h" includes additional attributes about the source
    and target. If NetworkX is installed (required for most methods), "graph" includes an
    NX representation of the transmission tree.

    Args:
        people (People): the sim.people object
    '''

    def __init__(self, people):

        # Pull out each of the attributes relevant to transmission
        attrs = {'age', 'date_exposed', 'date_symptomatic', 'date_tested', 'date_diagnosed', 'date_quarantined', 'date_severe', 'date_critical', 'date_known_contact', 'date_recovered'}

        self.n_days = people.t  # people.t should be set to the last simulation timestep in the output (since the Transtree is constructed after the people have been stepped forward in time)
        self.pop_size = len(people)

        # Include the basic line list
        self.infection_log = sc.dcp(people.infection_log)

        # Include the detailed transmission tree as well
        self.detailed = self.make_detailed(people)

        # Check for NetworkX
        nx = import_nx()
        if nx:
            self.graph = nx.DiGraph()

            for i in range(len(people)):
                d = {}
                for attr in attrs:
                    d[attr] = people[attr][i]
                self.graph.add_node(i, **d)

            # Next, add edges from linelist
            for edge in people.infection_log:
                self.graph.add_edge(edge['source'],edge['target'],date=edge['date'],layer=edge['layer'])

        return


    def __len__(self):
        '''
        The length of the transmission tree is the length of the line list,
        which should equal the number of infections.
        '''
        try:
            return len(self.infection_log)
        except:
            return 0


    @property
    def transmissions(self):
        """
        Iterable over edges corresponding to transmission events

        This excludes edges corresponding to seeded infections without a source
        """
        nx = import_nx()
        if nx:
            return nx.subgraph_view(self.graph, lambda x: x is not None).edges
        else:
            output = []
            for d in self.infection_log:
                if d['source'] is not None:
                    output.append([d['source'], d['target']])
            return output


    def make_detailed(self, people, reset=False):
        ''' Construct a detailed transmission tree, with additional information for each person '''
        # Reset to look like the line list, but with more detail
        detailed = [None]*self.pop_size

        for transdict in self.infection_log:

            # Pull out key quantities
            ddict  = sc.objdict(sc.dcp(transdict)) # For "detailed dictionary"
            source = ddict.source
            target = ddict.target
            ddict.s = sc.objdict() # Source
            ddict.t = sc.objdict() # Target

            # If the source is available (e.g. not a seed infection), loop over both it and the target
            if source is not None:
                stdict = {'s':source, 't':target}
            else:
                stdict = {'t':target}

            # Pull out each of the attributes relevant to transmission
            attrs = ['age', 'date_symptomatic', 'date_tested', 'date_diagnosed', 'date_quarantined', 'date_severe', 'date_critical', 'date_known_contact']
            for st,stind in stdict.items():
                for attr in attrs:
                    ddict[st][attr] = people[attr][stind]
            if source is not None:
                for attr in attrs:
                    if attr.startswith('date_'):
                        is_attr = attr.replace('date_', 'is_') # Convert date to a boolean, e.g. date_diagnosed -> is_diagnosed
                        ddict.s[is_attr] = ddict.s[attr] <= ddict['date'] # These don't make sense for people just infected (targets), only sources

                ddict.s.is_asymp   = np.isnan(people.date_symptomatic[source])
                ddict.s.is_presymp = ~ddict.s.is_asymp and ~ddict.s.is_symptomatic # Not asymptomatic and not currently symptomatic
            ddict.t['is_quarantined'] = ddict.t['date_quarantined'] <= ddict['date'] # This is the only target date that it makes sense to define since it can happen before infection

            detailed[target] = ddict

        return detailed


    def r0(self, recovered_only=False):
        """
        Return average number of transmissions per person

        This doesn't include seed transmissions. By default, it also doesn't adjust
        for length of infection (e.g. people infected towards the end of the simulation
        will have fewer transmissions because their infection may extend past the end
        of the simulation, these people are not included). If 'recovered_only=True'
        then the downstream transmissions will only be included for people that recover
        before the end of the simulation, thus ensuring they all had the same amount of
        time to transmit.
        """
        n_infected = []
        for i, node in self.graph.nodes.items():
            if i is None or np.isnan(node['date_exposed']) or (recovered_only and node['date_recovered']>self.n_days):
                continue
            n_infected.append(self.graph.out_degree(i))
        return np.mean(n_infected)


    def plot(self, *args, **kwargs):
        ''' Plot the transmission tree '''

        fig_args = kwargs.get('fig_args', dict(figsize=(16, 10)))

        ttlist = []
        for source_ind, target_ind in self.transmissions:
            ddict = self.detailed[target_ind]
            source = ddict.s
            target = ddict.t

            tdict = {}
            tdict['date'] =  ddict['date']
            tdict['layer'] =  ddict['layer']
            tdict['s_asymp'] =  np.isnan(source['date_symptomatic']) # True if they *never* became symptomatic
            tdict['s_presymp'] =  ~tdict['s_asymp'] and tdict['date']<source['date_symptomatic'] # True if they became symptomatic after the transmission date
            tdict['s_sev'] = source['date_severe'] < tdict['date']
            tdict['s_crit'] = source['date_critical'] < tdict['date']
            tdict['s_diag'] = source['date_diagnosed'] < tdict['date']
            tdict['s_quar'] = source['date_quarantined'] < tdict['date']
            tdict['t_quar'] = target['date_quarantined'] < tdict['date'] # What if the target was released from quarantine?
            ttlist.append(tdict)

        df = pd.DataFrame(ttlist).rename(columns={'date': 'Day'})
        df = df.loc[df['layer'] != 'seed_infection']

        df['Stage'] = 'Symptomatic'
        df.loc[df['s_asymp'], 'Stage'] = 'Asymptomatic'
        df.loc[df['s_presymp'], 'Stage'] = 'Presymptomatic'

        df['Severity'] = 'Mild'
        df.loc[df['s_sev'], 'Severity'] = 'Severe'
        df.loc[df['s_crit'], 'Severity'] = 'Critical'

        fig = pl.figure(**fig_args)
        i = 1;
        r = 2;
        c = 3

        def plot_quantity(key, title, i):
            dat = df.groupby(['Day', key]).size().unstack(key)
            ax = pl.subplot(r, c, i);
            dat.plot(ax=ax, legend=None)
            pl.legend(title=None)
            ax.set_title(title)

        to_plot = {
            'layer': 'Layer',
            'Stage': 'Source stage',
            's_diag': 'Source diagnosed',
            's_quar': 'Source quarantined',
            't_quar': 'Target quarantined',
            'Severity': 'Symptomatic source severity'
        }
        for i, (key, title) in enumerate(to_plot.items()):
            plot_quantity(key, title, i + 1)

        return fig


    def animate(self, *args, **kwargs):
        '''
        Animate the transmission tree.

        Args:
            animate    (bool):  whether to animate the plot (otherwise, show when finished)
            verbose    (bool):  print out progress of each frame
            markersize (int):   size of the markers
            sus_color  (list):  color for susceptibles
            fig_args   (dict):  arguments passed to pl.figure()
            axis_args  (dict):  arguments passed to pl.subplots_adjust()
            plot_args  (dict):  arguments passed to pl.plot()
            delay      (float): delay between frames in seconds
            font_size  (int):   size of the font
            colors     (list):  color of each person
            cmap       (str):   colormap for each person (if colors is not supplied)

        Returns:
            fig: the figure object
        '''

        # Settings
        animate = kwargs.get('animate', True)
        verbose = kwargs.get('verbose', False)
        msize = kwargs.get('markersize', 10)
        sus_color = kwargs.get('sus_color', [0.5, 0.5, 0.5])
        fig_args = kwargs.get('fig_args', dict(figsize=(24, 16)))
        axis_args = kwargs.get('axis_args', dict(left=0.10, bottom=0.05, right=0.85, top=0.97, wspace=0.25, hspace=0.25))
        plot_args = kwargs.get('plot_args', dict(lw=2, alpha=0.5))
        delay = kwargs.get('delay', 0.2)
        font_size = kwargs.get('font_size', 18)
        colors = kwargs.get('colors', None)
        cmap = kwargs.get('cmap', 'parula')
        pl.rcParams['font.size'] = font_size
        if colors is None:
            colors = sc.vectocolor(self.pop_size, cmap=cmap)

        # Initialization
        n = self.n_days + 1
        frames = [list() for i in range(n)]
        tests = [list() for i in range(n)]
        diags = [list() for i in range(n)]
        quars = [list() for i in range(n)]

        # Construct each frame of the animation
        for ddict in self.detailed:  # Loop over every person
            if ddict is None:
                continue # Skip the 'None' node corresponding to seeded infections

            frame = sc.objdict()
            tdq = sc.objdict()  # Short for "tested, diagnosed, or quarantined"
            target = ddict.t
            target_ind = ddict['target']

            if not np.isnan(ddict['date']): # If this person was infected

                source_ind = ddict['source'] # Index of the person who infected the target

                target_date = ddict['date']
                if source_ind is not None:  # Seed infections and importations won't have a source
                    source_date = self.detailed[source_ind]['date']
                else:
                    source_ind = 0
                    source_date = 0

                # Construct this frame
                frame.x = [source_date, target_date]
                frame.y = [source_ind, target_ind]
                frame.c = colors[source_ind]
                frame.i = True  # If this person is infected
                frames[int(target_date)].append(frame)

                # Handle testing, diagnosis, and quarantine
                tdq.t = target_ind
                tdq.d = target_date
                tdq.c = colors[int(target_ind)]
                date_t = target['date_tested']
                date_d = target['date_diagnosed']
                date_q = target['date_known_contact']
                if ~np.isnan(date_t) and date_t < n:
                    tests[int(date_t)].append(tdq)
                if ~np.isnan(date_d) and date_d < n:
                    diags[int(date_d)].append(tdq)
                if ~np.isnan(date_q) and date_q < n:
                    quars[int(date_q)].append(tdq)

            else:
                frame.x = [0]
                frame.y = [target_ind]
                frame.c = sus_color
                frame.i = False
                frames[0].append(frame)

        # Configure plotting
        fig = pl.figure(**fig_args)
        pl.subplots_adjust(**axis_args)
        ax = fig.add_subplot(1, 1, 1)

        # Create the legend
        ax2 = pl.axes([0.85, 0.05, 0.14, 0.9])
        ax2.axis('off')
        lcol = colors[0]
        na = np.nan  # Shorten
        pl.plot(na, na, '-', c=lcol, **plot_args, label='Transmission')
        pl.plot(na, na, 'o', c=lcol, markersize=msize, **plot_args, label='Source')
        pl.plot(na, na, '*', c=lcol, markersize=msize, **plot_args, label='Target')
        pl.plot(na, na, 'o', c=lcol, markersize=msize * 2, fillstyle='none', **plot_args, label='Tested')
        pl.plot(na, na, 's', c=lcol, markersize=msize * 1.2, **plot_args, label='Diagnosed')
        pl.plot(na, na, 'x', c=lcol, markersize=msize * 2.0, label='Known contact')
        pl.legend()

        # Plot the animation
        pl.sca(ax)
        for day in range(n):
            pl.title(f'Day: {day}')
            pl.xlim([0, n])
            pl.ylim([0, len(self)])
            pl.xlabel('Day')
            pl.ylabel('Person')
            flist = frames[day]
            tlist = tests[day]
            dlist = diags[day]
            qlist = quars[day]
            for f in flist:
                if verbose: print(f)
                pl.plot(f.x[0], f.y[0], 'o', c=f.c, markersize=msize, **plot_args)  # Plot sources
                pl.plot(f.x, f.y, '-', c=f.c, **plot_args)  # Plot transmission lines
                if f.i:  # If this person is infected
                    pl.plot(f.x[1], f.y[1], '*', c=f.c, markersize=msize, **plot_args)  # Plot targets
            for tdq in tlist: pl.plot(tdq.d, tdq.t, 'o', c=tdq.c, markersize=msize * 2, fillstyle='none')  # Tested; No alpha for this
            for tdq in dlist: pl.plot(tdq.d, tdq.t, 's', c=tdq.c, markersize=msize * 1.2, **plot_args)  # Diagnosed
            for tdq in qlist: pl.plot(tdq.d, tdq.t, 'x', c=tdq.c, markersize=msize * 2.0)  # Quarantine; no alpha for this
            pl.plot([0, day], [0.5, 0.5], c='k', lw=5)  # Plot the endless march of time
            if animate:  # Whether to animate
                pl.pause(delay)

        return fig


