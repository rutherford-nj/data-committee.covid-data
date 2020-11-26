"""
Library to get covid data for the 07070, BC, NJ, and US, then and display pretty plots and tables.

This is very overengineered and was done for s+g

Greg Recine <greg@gregrecine.com> Oct 18 2020
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import rpy2.robjects as robjects

from .defaults import SmoothingParams, RegionNames, DataFileNames, DefaultPopulations, set_plot_defaults


def std_dev(series, win, min_periods=1):
    """ Return rolling standard deviation

    :param series: pandas series
    :param win: rolling window
    :param min_periods: min periods to use (from start)
    :return: pandas series
    """
    return series.rolling(win).mean().rolling(win, min_periods=min_periods).std()


def sma(series, win):
    """ Return rolling simple moving average

    :param series: pandas series
    :param win: rolling window
    :return: pandas series
    """
    return series.rolling(win).mean()


def ewma(series, spn):
    """ Return rolling exponentialy weighted moving average

    :param series: pandas series
    :param spn: rolling window
    :return: pandas series
    """
    return series.ewm(span=spn).mean()


def wow(df):
    """ Return dataframe with day on week and week-over-week difference

    :param df: pandas dataframe ['Date', value to g=calc wow'
    :return: pandas dataframe
    """
    _df = df.copy()
    _df['day_of_week'] = _df['Date'].dt.weekday
    _df = _df.set_index('Date')
    _df['wow'] = _df.groupby('day_of_week').diff()
    return _df[['day_of_week', 'wow']]


def incidence(series, win):
    """ The occurrence of new cases of disease (COVID-19) in a population over a specific period of time.

    :param series: pandas series
    :param win: time period
    :return: pandas series
    """
    return series.rolling(win).sum()


class Settings:
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


class GetData:
    """
    Gets and massages town covid data
    """

    def __init__(self, settings=None):

        self.settings = Settings() if settings is None else settings

        # Get data from sources
        self.data_df_dict = self.get_data()

        # Pad data with zeros back to earliest date
        self.pad_date()

        regions = self.settings.regions

        # Do smoothing
        for key in regions:
            region = regions[key]
            self.data_df_dict[region] = self.do_smoothing(self.data_df_dict[region], region)

    def get_data(self):
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

    def pad_date(self):
        """Pad zeros back to the earliest date in all the files"""
        # noinspection PyArgumentList
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


class MakePlots:
    """
    Class that gets and massages town covid data

    Plot function definitions must start with plot_
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

    def rutherford_new_cases(self):
        """ Plot Rutherford local case numbers and -/+ 1 std

        :return: nothing, creates plot
        """

        # New cases for Rutherford
        def _plot_fn(ax):
            _df = self.covid_df['Rutherford']

            _df['New Cases std'] = std_dev(_df['New Cases'], self.sma_win)
            _df['New Cases sma'] = sma(_df['New Cases'], self.sma_win)

            cases_pos = _df['New Cases sma'] + _df['New Cases std']
            cases_neg = _df['New Cases sma'] - _df['New Cases std']

            ax.fill_between(_df['Date'], cases_neg, cases_pos, alpha=0.2, facecolor='#089FFF')
            ax.plot(_df['Date'], _df['New Cases sma'])
            ax.stem(_df['Date'], _df['New Cases'], linefmt='x:', markerfmt=' ', basefmt=' ')

            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_Rutherford',
                       'title': 'Rutherford New Covid Cases',
                       'legend': [str(self.sma_win) + 'd avg',
                                  'Uncertainty',
                                  'Daily Cases']
                       }
        self._make_plot(plot_config)

    def rutherford_total_cases(self):
        """ Plot Rutherford local total case numbers and -/+ 1 std

        :return: nothing, creates plot
        """

        # New cases for Rutherford
        def _plot_fn(ax):
            _df = self.covid_df['Rutherford']

            _df['Total Cases std'] = std_dev(_df['Total Cases'], self.sma_win)
            # _df['Total Cases sma'] = self.get_sma(_df['Total Cases'], self.sma_win)

            cases_pos = _df['Total Cases'] + _df['Total Cases std']
            cases_neg = _df['Total Cases'] - _df['Total Cases std']

            ax.fill_between(_df['Date'], cases_neg, cases_pos, alpha=0.2, facecolor='#089FFF')
            # ax.plot(_df['Date'], _df['Total Cases sma'])
            ax.plot(_df['Date'], _df['Total Cases'], 'b-')

            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'total_cases_Rutherford',
                       'title': 'Rutherford Total Covid Cases',
                       'legend': ['Total Cases',
                                  'Uncertainty']
                       }
        self._make_plot(plot_config)

    def new_cases_norm_sma(self):
        """ Plots new cases SMA across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        # All New Cases Scaled by SMA
        def _plot_fn(ax):
            # regions = ['US', 'NJ', 'Bergen', 'Counties', 'Rutherford']
            regions = ['US', 'NJ', 'Bergen', 'Rutherford']

            for region in regions:
                _df = self.covid_df[region]
                ax.plot(_df['Date'], sma(_df['New Cases / 100K'], self.sma_win))
            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_per_100K_' + str(self.sma_win) + 'd_SMA',
                       'title': 'New Cases / 100K residents -- ' + str(self.sma_win) + ' day average',
                       'legend': ['United States',
                                  'New Jersey',
                                  # 'Other Counties',
                                  'Bergen County',
                                  'Rutherford']
                       }
        self._make_plot(plot_config)

    def new_cases_norm_ewma(self):
        """ Plots new cases EWMA across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            # regions = ['US', 'NJ', 'Bergen', 'Counties', 'Rutherford']
            regions = ['US', 'NJ', 'Bergen', 'Rutherford']

            for region in regions:
                _df = self.covid_df[region]
                ax.plot(_df['Date'], sma(_df['New Cases / 100K'], self.ewma_spn))
            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_per_100K_' + str(self.ewma_spn) + 'd_EWMA',
                       'title': 'New Cases / 100K residents -- ' + str(self.ewma_spn) + ' day weighted average',
                       'legend': ['United States',
                                  'New Jersey',
                                  # 'Other Counties',
                                  'Bergen County',
                                  'Rutherford']
                       }
        self._make_plot(plot_config)

    def total_cases_norm(self):
        """ Plots total cases across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            # regions = ['US', 'NJ', 'Bergen', 'Counties', 'Rutherford']
            regions = ['US', 'NJ', 'Bergen', 'Rutherford']

            for region in regions:
                _df = self.covid_df[region]
                ax.plot(_df['Date'], _df['New Cases / 100K'])
            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'total_cases_per_100K',
                       'title': 'Total cases / 100K residents',
                       'legend': ['United States',
                                  'New Jersey',
                                  # 'Other Counties',
                                  'Bergen County',
                                  'Rutherford']
                       }
        self._make_plot(plot_config)

    def total_cases_norm_sma(self):
        """ Plots total cases across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            # regions = ['US', 'NJ', 'Bergen', 'Counties', 'Rutherford']
            regions = ['US', 'NJ', 'Bergen', 'Rutherford']

            for region in regions:
                _df = self.covid_df[region]
                ax.plot(_df['Date'], sma(_df['Total Cases / 100K'], self.sma_win))
            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'total_cases_per_100K_' + str(self.sma_win) + 'd_SMA',
                       'title': 'Total cases / 100K residents -- ' + str(self.sma_win) + ' day average',
                       'legend': ['United States',
                                  'New Jersey',
                                  # 'Other Counties',
                                  'Bergen County',
                                  'Rutherford']
                       }
        self._make_plot(plot_config)

    def new_cases_norm_sma_wow(self):
        """ Plots week over week cases across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            region_list = ['US', 'NJ', 'Bergen', 'Rutherford']
            # region_list = ['Rutherford']
            y_col = 'New Cases / 100K'
            for region in region_list:
                df = self.covid_df[region]
                wow_df = wow(df[['Date', y_col]])
                ax.plot(wow_df.index, sma(wow_df.wow, self.sma_win))
            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_week-over-week_' + str(self.sma_win) + 'avg',
                       'title': 'New Cases Week-over-Week: ' + str(self.sma_win) + ' avg',
                       'legend': ['US', 'NJ', 'Bergen', 'Rutherford']
                       }
        self._make_plot(plot_config)

    def new_cases_norm_sma_slope(self):
        """ Plots first deriv of new cases across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            region_list = ['US', 'NJ', 'Bergen', 'Rutherford']
            y_col = 'New Cases / 100K'
            for region in region_list:
                _df = self.covid_df[region]
                _df['slope'] = np.gradient(sma(_df[y_col], self.sma_win))
                _df['slope'] = sma(_df['slope'], win=7)
                ax.plot(_df.Date, _df.slope)
            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_per_100K_' + str(self.sma_win) + 'd_SMA_7d_slope',
                       'title': 'New Cases 7d avg slope of ' + str(self.sma_win) + ' avg/100K',
                       'legend': ['US', 'NJ', 'Bergen', 'Rutherford']
                       }
        self._make_plot(plot_config)

    def incidence(self):
        """ Plots two week sum of new cases, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            region_list = ['US', 'NJ', 'Bergen', 'Rutherford']
            y_col = 'New Cases / 100K'
            for region in region_list:
                _df = self.covid_df[region]
                ax.plot(_df['Date'], incidence(_df[y_col], 14))

            ax.axhline(10, color='green', linestyle=':')
            ax.axhline(50, color='orange', linestyle=':')
            ax.axhline(100, color='red', linestyle=':')

            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_per_100K_incindence',
                       'title': 'New Cases/100K 14d incidence (sum)',
                       'legend': ['US', 'NJ', 'Bergen', 'Rutherford']
                       }
        self._make_plot(plot_config)

    def trajectory(self):
        """ Plots trajectory slope of spline of 3d(?) avg of normalized new cases

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            region_list = ['US', 'NJ', 'Bergen', 'Rutherford']
            y_col = 'New Cases / 100K'
            for region in region_list:
                _df = self.covid_df[region].reset_index(drop=True)
                _df['incidence'] = incidence(_df[y_col], 14)
                _df['incidence_sma'] = sma(_df['incidence'], 3)

                x_val = _df.incidence_sma.dropna().index.tolist()
                y_val = _df.incidence_sma.dropna().values
                x_dates = _df[_df.index.isin(x_val)]['Date']

                r_x = robjects.FloatVector(x_val)
                r_y = robjects.FloatVector(y_val)

                r_smooth_spline = robjects.r['smooth.spline']  # extract R function
                spline_xy = r_smooth_spline(x=r_x, y=r_y, spar=0.5)
                spline = np.array(robjects.r['predict'](spline_xy, robjects.FloatVector(x_val)).rx2('y'))
                slope = np.gradient(spline)

                ax.plot(x_dates, slope)

            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_per_100K_3d_trajectory',
                       'title': 'New Cases/100K 3 Day trajectory',
                       'legend': ['US', 'NJ', 'Bergen', 'Rutherford']
                       }
        self._make_plot(plot_config)

    def _make_plot(self, config):
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


class MakeStats:
    """
    Class that gets and massages town covid data

    Stats function definitions must start with plot_
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

    def calc_stat_type_here(self):
        """ Calculate something good. Just do something stupid for now

        :return: nothing, calcs stat
        """

        # New cases for Rutherford
        def _stat_fn():
            _df = self.covid_df['Rutherford']

            _df['New Cases std'] = _df[str(self.sma_win) + 'd avg'].rolling(self.sma_win, min_periods=1).std()

            stat = _df['New Cases std'].iloc[-1]

            return stat

        table_config = {'stat_fn': _stat_fn,
                        'fname': 'sample_stat',
                        'title': 'Rutherford New Covid Cases Current Standard Dev',
                        'legend': ['Uncertainty']
                        }
        self._make_table(table_config)

    def _make_table(self, config=None):
        """ TODO: Print some tabular information of stats

        :param config: Dict TBD
        """

        config = {} if config is None else config

        if self._debug:
            print(config)
        else:
            pass
