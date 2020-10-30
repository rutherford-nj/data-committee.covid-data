"""
Get covid data for the 07070 and display pretty plots and tables.

This is very overengineered and was done for s+g

Greg Recine <greg@gregrecine.com> Oct 18 2020
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from cycler import cycler

# Set some default params for data smoothing
SmoothingParams = {
    # Simple moving average window
    'SMA_WIN': 14,

    # Exponentially Weighted Moving Average span
    'EWMA_SPAN': 14  # spn=29, com=14 | spn=15, com=7
}

# Set some default plots to make
MakeThesePlots = {
    'NEW_NORM_SMA': True,  # All New Cases Scaled by SMA
    'NEW_NORM_EWMA': False,  # All New Cases Scaled by EWMA
    'TOTAL_NORM': False,  # Total New Cases Scaled
    'TOTAL_NORM_SMA': True,  # Total New Cases Scaled by SMA
    'LOCAL_NEW': True,  # Rutherford new and total cases, unscaled
    'LOCAL_TOTAL': True  # Rutherford new and total cases, unscaled
}

RegionNames = {'US': 'United_States',
               'NJ': 'New_Jersey',
               'Bergen': 'Bergen_County',
               'Counties': 'NJ_Counties',
               'Rutherford': 'Rutherford'}

# The data we're importing default file names
DataFileNames = {RegionNames['US']: 'covid_tracking_us.csv',
                 RegionNames['NJ']: 'covid_tracking_nj.csv',
                 RegionNames['Counties']: 'nytimes_nj_counties.csv',
                 RegionNames['Rutherford']: 'rutherford_data.csv'}

# 2019 populations
DefaultPopulations = {RegionNames['US']: 328.2E6,
                      RegionNames['NJ']: 8.882E6,
                      RegionNames['Bergen']: 932202,
                      RegionNames['Rutherford']: 18303}
DefaultPopulations[RegionNames['Counties']] = (DefaultPopulations[RegionNames['NJ']]
                                               - DefaultPopulations[RegionNames['Bergen']])


def set_plot_defaults():
    """ Set up ploting defaults """

    # Black, Green, Orange, Blue
    color_palette = ['#000000', '#009E73', '#D55E00', '#0072B2']

    sns.set(rc={
        'axes.axisbelow': False,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        'axes.grid': True,
        'axes.labelcolor': 'dimgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.prop_cycle': cycler('color', color_palette),
        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        'text.color': 'dimgrey',
        'xtick.bottom': False,
        'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        'ytick.left': False,
        'ytick.right': False,
    }
    )

    sns.set_context("notebook", rc={"font.size": 16,
                                    "axes.titlesize": 20,
                                    "axes.labelsize": 18})


class CovidSettings:
    """ Settings for the runs go here """

    def __init__(self, settings=None):
        """

        :param settings:
            data_files: dict of file description and file name
            data_dir: where the data is located
            regions: list of region names
            population:  Rutherford 2019 population
            sma_win: sma window in days
            ewma_spn: ewma span in days
            debug: debug flag
        """

        # Set config to empty dict if not provided
        self.settings = {} if settings is None else settings

        # Set default settings
        self.settings['data_files'] = self.settings.get('data_files', DataFileNames)
        self.settings['data_dir'] = self.settings.get('data_dir', os.path.join(os.getcwd(), 'data', 'csv'))
        self.settings['population'] = self.settings.get('population', DefaultPopulations)
        self.settings['regions'] = self.settings.get('regions', RegionNames)
        self.settings['sma_win'] = self.settings.get('sma_win', SmoothingParams['SMA_WIN'])
        self.settings['ewma_spn'] = self.settings.get('ewma_spn', SmoothingParams['EWMA_SPAN'])
        self.settings['debug'] = self.settings.get('debug', False)

    @property
    def data_files(self):
        """Return data_files setting"""
        return self.settings['data_files']

    @property
    def data_dir(self):
        """Return data_dir setting"""
        return self.settings['data_dir']

    @property
    def population(self):
        """Return population setting"""
        return self.settings['population']

    @property
    def regions(self):
        """Return regions setting"""
        return self.settings['regions']

    @property
    def sma_win(self):
        """Return sma_win setting"""
        return self.settings['sma_win']

    @property
    def ewma_spn(self):
        """Return ewma_spn setting"""
        return self.settings['ewma_spn']

    @property
    def debug(self):
        """Return debug setting"""
        return self.settings['debug']


class CovidData:
    """
    Gets and massages town covid data
    """

    def __init__(self, settings=None):

        self.settings = CovidSettings() if settings is None else settings

        # Get data from sources
        self.data_df_dict = self._get_data()

        # Pad data with zeros back to earliest date
        self._pad_date()

        regions = self.settings.regions

        # Do smoothing
        for key in regions:
            region = regions[key]
            self.data_df_dict[region] = self.do_smoothing(self.data_df_dict[region], region)

    def _get_data(self):
        """ Grab data from the google sheet and pop into a pandas dataframe

        :return: df: dataframe of: Date, New Cases, Total Cases
        """

        data_dir = self.settings.data_dir
        data_files = self.settings.data_files
        regions = self.settings.regions

        data_df_dict = {}
        for key in data_files.keys():
            _df = pd.read_csv(os.path.join(data_dir, data_files[key]), parse_dates=[0])
            if key == regions['Counties']:
                _df['New Cases'] = _df.groupby('County').apply(lambda x: x['Total Cases'].diff()).reset_index(level=0,
                                                                                                              drop=True)
                _df.loc[_df['New Cases'].isna(), 'New Cases'] = _df['Total Cases']

                # Break out Bergen County
                data_df_dict[regions['Bergen']] = _df[_df.County == 'Bergen']

                # NJ ex Bergen County
                _df = _df[_df.County != 'Bergen'].groupby('Date').sum()
                _df.reset_index(inplace=True)

            data_df_dict[key] = _df.sort_values('Date')

        return data_df_dict

    def _pad_date(self):
        # Pad zeros back to the earliest date in all the files
        min_date = pd.Timestamp.today()

        regions = self.settings.regions

        for key in regions:
            _df = self.data_df_dict[regions[key]]
            min_date = min(min_date, _df['Date'].min())

        for key in regions:
            _df = self.data_df_dict[regions[key]]
            _df_min_dt = _df['Date'].min()

            if min_date != _df_min_dt:
                new_dates = pd.date_range(start=min_date, end=_df_min_dt - pd.DateOffset(1))
                _df = _df.append(pd.DataFrame({'Date': new_dates,
                                               'Total Cases': [0] * len(new_dates),
                                               'New Cases': [0] * len(new_dates)}))
                self.data_df_dict[regions[key]] = _df.sort_values('Date')

    def do_smoothing(self, covid_df, region):
        """
        Do n-day SMA and m-day EWMA smoothing
        :param region: data region
        :param covid_df: input data frame (date, total cases, new cases)
        :return: dataframe with new columns
        """
        sma_win = self.settings.sma_win
        ewma_spn = self.settings.ewma_spn
        population = self.settings.population[region]

        # Do smoothing with sma and ewma (unscaled and scaled)
        covid_df[str(sma_win) + 'd avg'] = covid_df['New Cases'].rolling(sma_win).mean()
        covid_df[str(ewma_spn) + 'd ewma'] = covid_df['New Cases'].ewm(span=ewma_spn).mean()

        covid_df['New Cases / 100K'] = covid_df['New Cases'] * (1.0E5 / population)
        covid_df[str(sma_win) + 'd avg / 100K'] = covid_df['New Cases / 100K'].rolling(sma_win).mean()
        covid_df[str(ewma_spn) + 'd ewma / 100K'] = covid_df['New Cases / 100K'].ewm(span=ewma_spn).mean()

        covid_df[str(sma_win) + 'd avg Total Cases'] = covid_df['Total Cases'].rolling(sma_win).mean()
        covid_df[str(ewma_spn) + 'd ewma Total'] = covid_df['Total Cases'].ewm(span=ewma_spn).mean()

        covid_df['Total Cases / 100K'] = covid_df['Total Cases'] * (1.0E5 / population)
        covid_df[str(sma_win) + 'd avg Total Cases / 100K'] = covid_df['Total Cases / 100K'].rolling(sma_win).mean()
        covid_df[str(ewma_spn) + 'd ewma Total Cases/ 100K'] = covid_df['Total Cases / 100K'].ewm(span=ewma_spn).mean()

        return covid_df


class CovidPlots:
    """
    Class that gets and massages town covid data
    """

    def __init__(self, covid_data):
        """

        :param covid_data: CovidData class
        """

        self.covid_data = covid_data

        set_plot_defaults()

        regions = covid_data.settings.regions

        self.covid_df = {}
        for key in regions:
            self.covid_df[key] = covid_data.data_df_dict[regions[key]]

    @property
    def _debug(self):
        """debug flag"""
        return self.covid_data.settings.debug

    @property
    def sma_win(self):
        """SMA Window"""
        return self.covid_data.settings.sma_win

    @property
    def ewma_spn(self):
        """EWMA Span"""
        return self.covid_data.settings.ewma_spn

    def local_new_cases(self):
        """ Plot Rutherford local case numbers and -/+ 1 std

        :return: nothing, creates plot
        """

        # New cases for Rutherford
        def _plot_fn(ax):
            _df = self.covid_df['Rutherford']

            _df['New Cases std'] = _df[str(self.sma_win) + 'd avg'].rolling(self.sma_win, min_periods=1).std()

            cases_pos = _df[str(self.sma_win) + 'd avg'] + _df['New Cases std']
            cases_neg = _df[str(self.sma_win) + 'd avg'] - _df['New Cases std']

            ax.plot(_df['Date'], _df[str(self.sma_win) + 'd avg'])
            ax.fill_between(_df['Date'], cases_neg, cases_pos, alpha=0.2, facecolor='#089FFF')

            # ax2 = ax.twinx()
            # ax2.plot(_df['Date'], _df['Total Cases'], color='black')
            # ax.plot(_df['Date'], _df['Total Cases'], color='black')

            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_Rutherford',
                       'title': 'Rutherford New Covid Cases',
                       'legend': ['Daily Cases',
                                  'Uncertainty']
                       }
        self.make_plot(plot_config)

    def local_total_cases(self):
        """ Plot Rutherford local total case numbers and -/+ 1 std

        :return: nothing, creates plot
        """

        # New cases for Rutheford
        def _plot_fn(ax):
            _df = self.covid_df['Rutherford']

            _df['Total Cases std'] = _df[str(self.sma_win) + 'd avg Total Cases'].rolling(self.sma_win,
                                                                                          min_periods=1).std()

            cases_pos = _df[str(self.sma_win) + 'd avg Total Cases'] + _df['Total Cases std']
            cases_neg = _df[str(self.sma_win) + 'd avg Total Cases'] - _df['Total Cases std']

            ax.plot(_df['Date'], _df[str(self.sma_win) + 'd avg Total Cases'])
            ax.fill_between(_df['Date'], cases_neg, cases_pos, alpha=0.2, facecolor='#089FFF')

            # ax2 = ax.twinx()
            # ax2.plot(_df['Date'], _df['Total Cases'], color='black')
            # ax.plot(_df['Date'], _df['Total Cases'], color='black')

            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'total_cases_Rutherford',
                       'title': 'Rutherford Total Covid Cases',
                       'legend': ['Total Cases',
                                  'Uncertainty']
                       }
        self.make_plot(plot_config)

    def new_cases_scaled_sma(self):
        """ Plots new cases SMA across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        # All New Cases Scaled by SMA
        def _new_cases_scaled_sma(ax):
            ax.plot(self.covid_df['US']['Date'], self.covid_df['US'][str(self.sma_win) + 'd avg / 100K'])
            ax.plot(self.covid_df['NJ']['Date'], self.covid_df['NJ'][str(self.sma_win) + 'd avg / 100K'])
            # ax.plot(self.df['Counties']['Date'], self.df['Counties'][str(self.sma_win) + 'd avg / 100K'])
            ax.plot(self.covid_df['Bergen']['Date'], self.covid_df['Bergen'][str(self.sma_win) + 'd avg / 100K'])
            ax.plot(self.covid_df['Rutherford']['Date'],
                    self.covid_df['Rutherford'][str(self.sma_win) + 'd avg / 100K'])

            return ax

        plot_config = {'plot_fn': _new_cases_scaled_sma,
                       'fname': 'new_cases_per_100K_' + str(self.sma_win) + 'd_SMA',
                       'title': 'New Cases / 100K residents -- ' + str(self.sma_win) + ' day average',
                       'legend': ['United States',
                                  'New Jersey',
                                  # 'Other Counties',
                                  'Bergen County',
                                  'Rutherford']
                       }
        self.make_plot(plot_config)

    def new_cases_scaled_ewma(self):
        """ Plots new cases EWMA across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _new_cases_scaled_ewma(ax):
            ax.plot(self.covid_df['US']['Date'], self.covid_df['US'][str(self.ewma_spn) + 'd ewma / 100K'])
            ax.plot(self.covid_df['NJ']['Date'], self.covid_df['NJ'][str(self.ewma_spn) + 'd ewma / 100K'])
            # ax.plot(self.df['Counties']['Date'], self.df['Counties'][str(self.ewma_spn) + 'd ewma / 100K'])
            ax.plot(self.covid_df['Bergen']['Date'], self.covid_df['Bergen'][str(self.ewma_spn) + 'd ewma / 100K'])
            ax.plot(self.covid_df['Rutherford']['Date'],
                    self.covid_df['Rutherford'][str(self.ewma_spn) + 'd ewma / 100K'])

            return ax

        plot_config = {'plot_fn': _new_cases_scaled_ewma,
                       'fname': 'new_cases_per_100K_' + str(self.ewma_spn) + 'd_EWMA',
                       'title': 'New Cases / 100K residents -- ' + str(self.ewma_spn) + ' day weighted average',
                       'legend': ['United States',
                                  'New Jersey',
                                  # 'Other Counties',
                                  'Bergen County',
                                  'Rutherford']
                       }
        self.make_plot(plot_config)

    def total_cases_scaled(self):
        """ Plots total cases across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _total_cases_scaled(ax):
            ax.plot(self.covid_df['US']['Date'], self.covid_df['US']['Total Cases / 100K'])
            ax.plot(self.covid_df['NJ']['Date'], self.covid_df['NJ']['Total Cases / 100K'])
            # ax.plot(self.df['Counties']['Date'], self.df['Counties']['Total Cases / 100K'])
            ax.plot(self.covid_df['Bergen']['Date'], self.covid_df['Bergen']['Total Cases / 100K'])
            ax.plot(self.covid_df['Rutherford']['Date'], self.covid_df['Rutherford']['Total Cases / 100K'])

            return ax

        plot_config = {'plot_fn': _total_cases_scaled,
                       'fname': 'total_cases_per_100K',
                       'title': 'Total cases / 100K residents',
                       'legend': ['United States',
                                  'New Jersey',
                                  # 'Other Counties',
                                  'Bergen County',
                                  'Rutherford']
                       }
        self.make_plot(plot_config)

    def total_cases_scaled_sma(self):
        """ Plots total cases across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _total_cases_scaled_sma(ax):
            ax.plot(self.covid_df['US']['Date'], self.covid_df['US'][str(self.sma_win) + 'd avg Total Cases / 100K'])
            ax.plot(self.covid_df['NJ']['Date'], self.covid_df['NJ'][str(self.sma_win) + 'd avg Total Cases / 100K'])
            # ax.plot(df['Counties']['Date'], self.df['Counties'][str(self.sma_win) + 'd avg Total Cases / 100K'])
            ax.plot(self.covid_df['Bergen']['Date'],
                    self.covid_df['Bergen'][str(self.sma_win) + 'd avg Total Cases / 100K'])
            ax.plot(self.covid_df['Rutherford']['Date'],
                    self.covid_df['Rutherford'][str(self.sma_win) + 'd avg Total Cases / 100K'])

            return ax

        plot_config = {'plot_fn': _total_cases_scaled_sma,
                       'fname': 'total_cases_per_100K_' + str(self.sma_win) + 'd_SMA',
                       'title': 'Total cases / 100K residents -- ' + str(self.sma_win) + ' day average',
                       'legend': ['United States',
                                  'New Jersey',
                                  # 'Other Counties',
                                  'Bergen County',
                                  'Rutherford']
                       }
        self.make_plot(plot_config)

    def do_tables(self, config=None):
        """ TODO: Print some tabular information of stats

        :param config: Dict TBD
        """

        config = {} if config is None else config

        if self._debug:
            print('stats here')
        else:
            pass

    def make_plot(self, config):
        """ Plot town cases and averages

        :param config: dict of
                - x_col: x columns (str)
                - y_cols: list of y columns
                - plot_args: list of args for plot
                - region: area we are plotting
                - fname: filename suffux
                - title: chart title (str)
        """

        fig = plt.figure(figsize=(13, 9))
        ax = fig.add_subplot(1, 1, 1)

        config['plot_fn'](ax)

        ax.set_ylabel('# of cases', fontsize=20)
        my_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_formatter(my_fmt)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.title(config['title'], fontsize=24)
        if config.get('legend', None) is not None:
            plt.legend(config['legend'], fontsize=16)
        plt.grid()

        if self._debug:
            plt.show()
        else:
            plt.savefig(config['fname'] + ".svg", format="svg")
            plt.close()


def main():
    """
    Get data and output useful stuff
    """

    # Init covid class
    # TODO: Make an argument
    sma_win = SmoothingParams['SMA_WIN']
    ewma_span = SmoothingParams['EWMA_SPAN']
    settings = {
        'sma_win': sma_win,
        'ewma_spn': ewma_span,
        'debug': False
    }

    # Lazy way to turn plots on and off (can be args later)
    # TODO: Make arguments
    do_new_norm_sma = MakeThesePlots['NEW_NORM_SMA']
    do_new_norm_ewma = MakeThesePlots['NEW_NORM_EWMA']
    do_totals_norm = MakeThesePlots['TOTAL_NORM']
    do_totals_norm_sma = MakeThesePlots['TOTAL_NORM_SMA']
    do_local_new_cases = MakeThesePlots['LOCAL_NEW']
    do_local_total_cases = MakeThesePlots['LOCAL_TOTAL']

    # Set up the run
    covid_settings = CovidSettings(settings)

    # Get the data
    covid_data = CovidData(covid_settings)

    # Dir to write output to (assuming root is where this script is)
    os.chdir('docs')

    # Set up plotting and plot away if set
    plot_data = CovidPlots(covid_data)

    if do_new_norm_sma:
        plot_data.new_cases_scaled_sma()

    if do_new_norm_ewma:
        plot_data.new_cases_scaled_ewma()

    if do_totals_norm:
        plot_data.total_cases_scaled()

    if do_totals_norm_sma:
        plot_data.total_cases_scaled_sma()

    if do_local_new_cases:
        plot_data.local_new_cases()

    if do_local_total_cases:
        plot_data.local_total_cases()


if __name__ == "__main__":
    main()
