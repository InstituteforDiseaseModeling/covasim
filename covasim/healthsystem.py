'''
Perform modeling of health systems capacity.
'''

import datetime as dt
import pylab as pl
import sciris as sc

__all__ = ['make_hspars', 'HealthSystem']


def make_hspars():
    '''
    Make defaults for health system parameters. Estimates from:
        https://docs.google.com/document/d/1fIs2kCuu33tTCpbHQ-0YfvqXP4bSrqg-/edit#heading=h.gjdgxs
    '''

    hsp = sc.objdict() # This is a dict but allows access via e.g. hsp.icu instead of hsp['icu']
    hsp.symptomatic  = 0.5  # Fraction of cases that are symptomatic
    hsp.hospitalized = 0.25 # Fraction of sympotmatic cases that require hospitalization
    hsp.icu          = 0.08 # Fraction of symptomatic cases that require ICU
    hsp.mild_dur     = 11   # Days of a moderate stay -- from pl.mean([7.9, 13.4])
    hsp.severe_dur   = 17   # Days of a severe stay -- from pl.mean([12.5, 21.2])
    hsp.aac_frac     = 0.5  # Fraction of time in severe cases that stay in an AAC bed
    hsp.delay        = 5    # Days between being infected and being hospitalized

    return hsp


class HealthSystem(sc.prettyobj):
    '''
    Class for storing, analyzing, a plotting health systems data.

    Data are assumed to come from Covasim and be of the format:
        data[result_type][scenario_name][best,low,high] = time series
        e.g.
        data.cum_exposed.baseline.best = [0, 1, 1, 2, 3, 5, 10, 13 ...]
    '''

    def __init__(self, data=None, filename=None, hspars=None, startday=0):
        ''' Initialize the object '''
        if filename is not None:
            if data is not None:
                raise ValueError(f'You can supply data or a filename, but what am I supposed to do with both?')
            data = sc.loadobj(filename)
        if hspars is None:
            hspars = make_hspars()
        self.data = data
        self.hspars = hspars
        self.results = None
        self.startday = startday
        return


    def parse_data(self, datakey=None, datatype=None):
        '''
        Ensure the data object has the right structure, and store the keys in the object.
        '''

        # Choose the data type -- by default, cumulative exposures
        if datakey is None:
            datakey = 'cum_exposed' # TODO: make this less hard-coded?
        if datatype is None:
            datatype = 'cumulative'
        self.datakey = datakey
        self.datatype = datatype

        # Check that the data is a dict of results types
        D = self.data # Shortcut
        if not isinstance(D, dict):
            raise TypeError(f'Data must be dict with keys for different results, but you supplied {type(D)}')

        # ...and then a dict of scenarios
        self.datakeys = list(D.keys())
        if self.datakey not in self.datakeys:
            raise KeyError(f'Could not find supplied datakey {self.datakey} in supplied datakeys {self.datakeys}')
        dk0 = self.datakeys[0] # For "data key 0"
        if not isinstance(D[dk0], dict):
            raise TypeError(f'The second level in the data must also be a dict, but you supplied {type(D[dk0])}')

        # ...and then a dict of best, high, low
        self.scenkeys = list(D[dk0].keys())
        sk0 = self.scenkeys[0]
        if not isinstance(D[dk0][sk0], dict):
            raise TypeError(f'The third level in the data must also be a dict, but you supplied {type(D[dk0][sk0])}')

        # ...and a numeric array
        self.blh = ['best', 'low', 'high']
        if not all([(key in D[dk0][sk0]) for key in self.blh]):
            raise ValueError(f'The required keys {self.blh} could not be found in {D[dk0][sk0].keys()}')
        if not sc.checktype(D[dk0][sk0].best, 'arraylike'):
            raise TypeError(f'Was expecting a numeric array, but got {type(D[dk0][sk0].best)}')

        # Figure out how many points are in this thing
        self.npts = len(D[dk0][sk0].best)

        # Store labels
        self.scenlabels = {scenkey:D[dk0][scenkey].name for scenkey in self.scenkeys}

        return


    def init_results(self):
        '''
        Initialize the results structure, e.g.:
            self.beds.aac.baseline.best = time series
        '''
        self.reskeys = ['aac', 'icu', 'total']
        self.reslabels = ['Adult acute beds', 'ICU beds', 'Total beds']
        self.beds = sc.objdict()
        for reskey in self.reskeys:
            self.beds[reskey] = sc.objdict()
            for scenkey in self.scenkeys:
                self.beds[reskey][scenkey] = sc.objdict()
                for blh in self.blh:
                    self.beds[reskey][scenkey][blh] = pl.zeros(self.npts)
        return


    # def safet(self, t):
    #     ''' Ensure the given timepoint is "safe", not off the end of the array '''
    #     return np.minimum(self.npts, t)


    def process_ts(self, ts):
        ''' The meat of the class -- convert an input time series into beds '''

        # Define a function to stop time points from going off the end of the array
        tlim = lambda t: pl.minimum(self.npts, t) # Short for "time limit"

        # Housekeeping
        hsp = self.hspars # Shorten since used a lot
        beds = sc.objdict() # To make in one step: make(keys=self.reskeys, vals=pl.zeros(self.npts))
        for reskey in self.reskeys:
            beds[reskey] = pl.zeros(self.npts)

        # If cumulative, take the difference to get the change at each timepoint
        if self.datatype == 'cumulative':
            ts = pl.diff(ts)

        # Actually process the time series -- where all the logic is, loop over each time point and update beds required
        for t,val in enumerate(ts):

            # Precompute results
            sympt         = val * hsp.symptomatic              # Find how many symptomatic people there are
            hosp          = sympt * hsp.hospitalized           # How many require hospitalization
            icu           = sympt * hsp.icu                    # How many will require ICU beds
            mild          = hosp - icu                         # Non-ICU patients are mild
            tstart_aac    = t + hsp.delay                      # When adult acute beds start being used
            tstop_aac     = tstart_aac + hsp.mild_dur          # When adult acute beds are no longer needed
            icu_in_aac    = round(hsp.severe_dur*hsp.aac_frac) # Days an ICU patient spends in AAC
            icu_in_icu    = hsp.severe_dur - icu_in_aac        # ...and in ICU
            tstop_pre_icu = tstart_aac + icu_in_aac            # When they move from AAC to ICU
            tstop_icu     = tstop_pre_icu + icu_in_icu         # When they leave ICU

            # Compute actual results
            beds.aac[tlim(tstart_aac):tlim(tstop_aac)]     += mild # Add mild patients to AAC
            beds.aac[tlim(tstart_aac):tlim(tstop_pre_icu)] += icu  # Add pre-ICU ICU patients
            beds.icu[tlim(tstop_pre_icu):tlim(tstop_icu)]  += icu  # Add ICU patients

        beds.total = beds.aac + beds.icu # Compute total results

        return beds


    def analyze(self):
        ''' Analyze the data and project resource needs -- all logic is in process_ts '''

        self.parse_data() # Make sure the data has the right structure
        self.init_results() # Create the ersults object

        for scenkey in self.scenkeys:
            for blh in self.blh:
                this_ts = self.data[self.datakey][scenkey][blh]
                thesebeds = self.process_ts(this_ts)
                for reskey in self.reskeys:
                    self.beds[reskey][scenkey][blh] = thesebeds[reskey]

        return


    def plot(self, do_save=None, fig_args=None, plot_args=None, scatter_args=None, fill_args=None,
             axis_args=None, font_size=None, font_family=None, use_grid=True, do_show=True, verbose=None):
        '''
        Plotting, copied from run_cdc_scenarios.

        Args:
            do_save (bool/str):  whether or not to save the figure, if so, to this filename
            fig_args (dict):     options for styling the figure (e.g. size)
            plot_args (dict):    likewise, for the plot (e.g., line thickness)
            scatter_args (dict): likewise, for scatter points (used for data)
            fill_args (dict):    likewise, for uncertainty bounds (e.g. alpha)
            axis_args (dict):    likewise, for axes (e.g. margins)
            font_size (int):     overall figure font size
            font_family (str):   what font to use (must exist on your system!)
            use_grid (bool):     whether or not to plot gridlines on the plot
            verbose (bool):      whether or not to print extra output

        Returns:
            fig: a matplotlib figure object
        '''

        if fig_args     is None: fig_args     = {'figsize':(16,12)}
        if plot_args    is None: plot_args    = {'lw':3, 'alpha':0.7}
        if scatter_args is None: scatter_args = {'s':150, 'marker':'s'}
        if axis_args    is None: axis_args    = {'left':0.1, 'bottom':0.05, 'right':0.9, 'top':0.97, 'wspace':0.2, 'hspace':0.25}
        if fill_args    is None: fill_args    = {'alpha': 0.2}
        if font_size    is None: font_size    = 18
        if font_family  is None: font_family  = 'Proxima Nova'

        fig = pl.figure(**fig_args)
        pl.subplots_adjust(**axis_args)
        pl.rcParams['font.size'] = font_size
        pl.rcParams['font.family'] = font_family

        xmin = self.startday
        tvec = xmin + pl.arange(self.npts) # TODO: fix! With dates!

        for rk,reskey in enumerate(self.reskeys):
            pl.subplot(len(self.reskeys),1,rk+1)

            resdata = self.beds[reskey]

            for scenkey, scendata in resdata.items():
                pl.fill_between(tvec, scendata.low, scendata.high, **fill_args)
                # pl.plot(tvec, scendata.low, linestyle='--', **plot_args)
                # pl.plot(tvec, scendata.high, linestyle='--', **plot_args)
                pl.plot(tvec, scendata.best, label=self.scenlabels[scenkey], **plot_args)

                if rk == 0:
                    pl.legend()

                pl.title(self.reslabels[rk])
                # sc.setylim()
                pl.grid(True)

                # Set x-axis
                xmax = xmin + self.npts # TODO: fix!!!
                pl.gca().set_xticks(pl.arange(xmin, xmax+1, 7))
                xt = pl.gca().get_xticks()
                lab = []
                for t in xt:
                    tmp = dt.datetime(2020, 1, 1) + dt.timedelta(days=int(t)) # + pars['day_0']
                    lab.append( tmp.strftime('%B %d') )
                pl.gca().set_xticklabels(lab)
                sc.commaticks(axis='y')

        if do_show:
            pl.show()

        return fig


def run_healthsystem(doplot=True, dosave=False, *args, **kwargs):
    ''' Shortcut for running a health system analysis '''
    hsys = HealthSystem(*args, **kwargs)
    hsys.analyze()
    if doplot:
        hsys.plot()
    return hsys

