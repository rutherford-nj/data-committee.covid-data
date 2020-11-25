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

    def plot_local_new_cases(self):
        """ Plot Rutherford local case numbers and -/+ 1 std

        :return: nothing, creates plot
        """

        # New cases for Rutherford
        def _plot_fn(ax):
            _df = self.covid_df['Rutherford']

            _df['New Cases std'] = _df['New Cases'].rolling(self.sma_win).mean().rolling(self.sma_win, min_periods=1).std()

            cases_pos = _df['New Cases'].rolling(self.sma_win).mean() + _df['New Cases std']
            cases_neg = _df['New Cases'].rolling(self.sma_win).mean() - _df['New Cases std']

            ax.fill_between(_df['Date'], cases_neg, cases_pos, alpha=0.2, facecolor='#089FFF')
            ax.plot(_df['Date'], _df['New Cases'].rolling(self.sma_win).mean())
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

    def plot_local_total_cases(self):
        """ Plot Rutherford local total case numbers and -/+ 1 std

        :return: nothing, creates plot
        """

        # New cases for Rutherford
        def _plot_fn(ax):
            _df = self.covid_df['Rutherford']

            _df['Total Cases std'] = _df['Total Cases'].rolling(self.sma_win).mean().rolling(self.sma_win, min_periods=1).std()

            cases_pos = _df['Total Cases'] + _df['Total Cases std']
            cases_neg = _df['Total Cases'] - _df['Total Cases std']

            ax.fill_between(_df['Date'], cases_neg, cases_pos, alpha=0.2, facecolor='#089FFF')
            # ax.plot(_df['Date'], _df['Total Cases'].rolling(self.sma_win).mean())
            ax.plot(_df['Date'], _df['Total Cases'], 'b-')

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
        self._make_plot(plot_config)

    def plot_new_cases_scaled_sma(self):
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
        self._make_plot(plot_config)

    def plot_new_cases_scaled_ewma(self):
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
        self._make_plot(plot_config)

    def plot_total_cases_scaled(self):
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
        self._make_plot(plot_config)

    def plot_total_cases_scaled_sma(self):
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
        self._make_plot(plot_config)

    def plot_new_cases_wow_scaled_sma(self):
        """ Plots week over week cases across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            region_list = ['US', 'NJ', 'Bergen', 'Rutherford']
            # region_list = ['Rutherford']
            y_col = 'New Cases / 100K'
            for region in region_list:
                df = self.covid_df[region]
                data = self.wow(df[['Date',y_col]])
                y_data = data.wow.rolling(self.sma_win).mean()
                x_data = y_data.index
                ax.plot(x_data, y_data)

            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_week-over-week_'+str(self.sma_win)+'avg',
                       'title': 'New Cases Week-over-Week: '+str(self.sma_win)+' avg',
                       'legend': ['US', 'NJ', 'Bergen', 'Rutherford']
                       }
        self._make_plot(plot_config)

    def plot_new_cases_slope_scaled_sma(self):
        """ Plots first deriv of new cases across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            region_list = ['US', 'NJ', 'Bergen', 'Rutherford']
            # region_list = ['US']
            y_col = str(self.sma_win) + 'd avg / 100K'
            for region in region_list:
                df = self.covid_df[region]
                data = df[['Date', y_col]].copy()
                data['slope'] = np.gradient(data[y_col])
                data['slope'] = data['slope'].rolling(7).mean()
                y_data = data.slope
                x_data = data.Date
                ax.plot(x_data, y_data)

            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_per_100K_' + str(self.sma_win) + '_SMA_7d_slope',
                       'title': 'New Cases 7d slope of ' + str(self.sma_win) + ' avg/100K',
                       'legend': ['US', 'NJ', 'Bergen', 'Rutherford']
                       }
        self._make_plot(plot_config)

    def plot_new_cases_scaled_sum(self):
        """ Plots two week sum of new cases, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            region_list = ['US', 'NJ', 'Bergen', 'Rutherford']
            # region_list = ['US']
            y_col = 'New Cases / 100K'
            # y_max = 0.0
            for region in region_list:
                _df = self.covid_df[region]
                ax.plot(_df['Date'], _df[y_col].rolling(14).sum())
                # y_max = max(y_max,_df[y_col].rolling(14).sum().max())

            ax.axhline(10, color='green', linestyle=':')
            ax.axhline(50, color='orange', linestyle=':')
            ax.axhline(100, color='red', linestyle=':')

            # ax.axhspan(0, 10, alpha=0.2, facecolor='green')
            # ax.axhspan(10, 50, alpha=0.2, facecolor='yellow')
            # ax.axhspan(50, 100, alpha=0.2, facecolor='orange')
            # ax.axhspan(100, y_max, alpha=0.2, facecolor='red')

            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_per_100K_sum',
                       'title': 'New Cases/100K 14d sum',
                       'legend': ['US', 'NJ', 'Bergen', 'Rutherford']
                       }
        self._make_plot(plot_config)

    def plot_new_cases_scaled_trajectory(self):
        """ Plots trajectory slope of spline of 3d(?) avg of normalized new cases

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            region_list = ['US', 'NJ', 'Bergen', 'Rutherford']
            # region_list = ['US']
            y_col = 'New Cases / 100K'
            # y_max = 0.0
            for region in region_list:
                _df = self.covid_df[region].reset_index(drop=True)
                _df['incidence'] = _df[y_col].rolling(14).sum()
                _df['incidence_3dAvg'] = _df['incidence'].rolling(3).mean()

                x_val = _df.incidence_3dAvg.dropna().index.tolist()
                y_val = _df.incidence_3dAvg.dropna().values
                x_dates = _df[_df.index.isin(x_val)]['Date']

                r_x = robjects.FloatVector(x_val)
                r_y = robjects.FloatVector(y_val)

                r_smooth_spline = robjects.r['smooth.spline']  # extract R function
                spline_xy = r_smooth_spline(x=r_x, y=r_y)
                spline = np.array(robjects.r['predict'](spline_xy, robjects.FloatVector(x_val)).rx2('y'))
                slope = np.gradient(spline)

                ax.plot(x_dates, slope)

            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_per_100K_3d_trajectory',
                       'title': 'New Cases/100K 3 Daytrajectory',
                       'legend': ['US', 'NJ', 'Bergen', 'Rutherford']
                       }
        self._make_plot(plot_config)

    def wow(self, df):
        _df = df.copy()
        _df['day_of_week'] = _df['Date'].dt.weekday
        _df = _df.set_index('Date')
        _df['wow'] = _df.groupby('day_of_week').diff()
        return _df[['day_of_week','wow']]

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
