"""
Library to get covid data for the 07070, BC, NJ, and US, then and display pretty plots and tables.

This is very overengineered and was done for s+g

Greg Recine <greg@gregrecine.com> Oct 18 2020
"""
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lib.defaults as defaults

from lib.utils import std_dev, sma, wow, ewma, incidence, smooth_slope


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
        self.settings['data_files'] = self.settings.get('data_files', defaults.DataFileNames)
        self.settings['data_dir'] = self.settings.get('data_dir', os.path.join(os.getcwd(), 'data', 'csv'))
        self.settings['population'] = self.settings.get('population', defaults.DefaultPopulations)
        self.settings['regions'] = self.settings.get('regions', defaults.RegionNames)
        self.settings['sma_win'] = self.settings.get('sma_win', defaults.SmoothingParams['SMA_WIN'])
        self.settings['ewma_spn'] = self.settings.get('ewma_spn', defaults.SmoothingParams['EWMA_SPAN'])
        self.settings['tail_days'] = self.settings.get('tail_days', defaults.LastNDays)
        self.settings['debug'] = self.settings.get('debug', False)

    @property
    def tail_days(self):
        """Return data_files setting"""
        return self.settings['tail_days']

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

    If initialized with ala_cart=True, you get to collect and modify the data yourself,
    otherwise it's done for you the way that is best for default running
    """

    def __init__(self, settings=None, ala_cart=False):

        self.settings = Settings() if settings is None else settings

        if not ala_cart:
            # Get data from sources
            self.data_df_dict = self.get_data()

            # Pad data with zeros back to earliest date
            self.pad_date()

            regions = self.settings.regions

            # Get normalized cases (per 100K pop)
            for key in regions:
                region = regions[key]
                self.data_df_dict[region] = self.per_capita(self.data_df_dict[region], region)

    def get_data(self):
        """ Grab data from the csv files and pop into a pandas dataframe

        :return: df: dataframe of: Date, New Cases, Total Cases
        """

        data_dir = self.settings.data_dir
        data_files = self.settings.data_files
        regions = self.settings.regions

        data_df_dict = {}
        for key in data_files.keys():
            _df = pd.read_csv(os.path.join(data_dir, data_files[key]), parse_dates=[0])
            if key != regions['Rutherford']:
                _df['New Cases'] = _df['Total Cases'].diff().reset_index(level=0, drop=True)
                _df.loc[_df['New Cases'].isna(), 'New Cases'] = _df['Total Cases']

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

    def per_capita(self, covid_df, region):
        """
        Normalize new and total cases per capita (100K)
        :param region: data region
        :param covid_df: input data frame (date, total cases, new cases)
        :return: dataframe with new columns
        """
        population = self.settings.population[region]

        covid_df['New Cases / 100K'] = covid_df['New Cases'] * (1.0E5 / population)

        covid_df['Total Cases / 100K'] = covid_df['Total Cases'] * (1.0E5 / population)

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

        defaults.plot_defaults()

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

    def rutherford_new_cases_reg(self):
        """ Plot Rutherford local case numbers and -/+ 1 std with regularized fitted curve

        :return: nothing, creates plot
        """

        # New cases for Rutherford
        def _plot_fn(ax):
            _df = self.covid_df['Rutherford']

            _df['New Cases std'] = std_dev(_df['New Cases'], self.sma_win)
            _df['New Cases sma'] = sma(_df['New Cases'], self.sma_win)

            y_col = 'New Cases'
            _x_dates, _slope, spline = smooth_slope(_df, y_col, inc_win=self.sma_win, sma_win=3, spar=0.5)
            spline = np.pad(spline, (len(_df) - len(spline), 0), 'constant', constant_values=0)
            _df['New Cases fitted'] = spline / self.sma_win

            cases_pos = _df['New Cases sma'] + _df['New Cases std']
            cases_neg = _df['New Cases sma'] - _df['New Cases std']

            ax.fill_between(_df['Date'], cases_neg, cases_pos, alpha=0.2, facecolor='#089FFF')
            ax.plot(_df['Date'], _df['New Cases fitted'])
            ax.plot(_df['Date'], _df['New Cases'], 'bo', mfc='none')

            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_Rutherford_fitted_' + str(self.sma_win) + 'd',
                       'title': 'Rutherford New Covid Cases',
                       'legend': [str(self.sma_win) + 'd avg fitted',
                                  'Daily Cases',
                                  'Uncertainty']
                       }
        self._make_plot(plot_config)

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_per_100K_3d_trajectory',
                       'title': 'New Cases/100K 3 Day trajectory',
                       'legend': ['US', 'NJ', 'Bergen', 'Rutherford']
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
            regions = defaults.PlotRegions

            for region in regions:
                _df = self.covid_df[region]
                ax.plot(_df['Date'], sma(_df['New Cases / 100K'], self.sma_win))
            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_per_100K_' + str(self.sma_win) + 'd_SMA',
                       'title': 'New Cases / 100K residents -- ' + str(self.sma_win) + ' day average',
                       'legend': defaults.PlotRegions
                       }
        self._make_plot(plot_config)

    def new_cases_norm_ewma(self):
        """ Plots new cases EWMA across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            regions = defaults.PlotRegions

            for region in regions:
                _df = self.covid_df[region]
                ax.plot(_df['Date'], ewma(_df['New Cases / 100K'], self.ewma_spn))
            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_per_100K_' + str(self.ewma_spn) + 'd_EWMA',
                       'title': 'New Cases / 100K residents -- ' + str(self.ewma_spn) + ' day weighted average',
                       'legend': defaults.PlotRegions
                       }
        self._make_plot(plot_config)

    def total_cases_norm(self):
        """ Plots total cases across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            regions = defaults.PlotRegions

            for region in regions:
                _df = self.covid_df[region]
                ax.plot(_df['Date'], _df['New Cases / 100K'])
            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'total_cases_per_100K',
                       'title': 'Total cases / 100K residents',
                       'legend': defaults.PlotRegions
                       }
        self._make_plot(plot_config)

    def total_cases_norm_sma(self):
        """ Plots total cases across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            regions = defaults.PlotRegions

            for region in regions:
                _df = self.covid_df[region]
                ax.plot(_df['Date'], sma(_df['Total Cases / 100K'], self.sma_win))
            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'total_cases_per_100K_' + str(self.sma_win) + 'd_SMA',
                       'title': 'Total cases / 100K residents -- ' + str(self.sma_win) + ' day average',
                       'legend': defaults.PlotRegions
                       }
        self._make_plot(plot_config)

    def new_cases_norm_sma_wow(self):
        """ Plots week over week cases across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            regions = defaults.PlotRegions
            y_col = 'New Cases / 100K'
            for region in regions:
                _df = self.covid_df[region]
                wow_df = wow(_df[['Date', y_col]])
                ax.plot(wow_df.index, sma(wow_df.wow, self.sma_win))

            ax.axhline(0, color='red', linestyle=':')

            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_week-over-week_' + str(self.sma_win) + 'avg',
                       'title': 'New Cases Week-over-Week: ' + str(self.sma_win) + ' avg',
                       'legend': defaults.PlotRegions
                       }
        self._make_plot(plot_config)

    def new_cases_norm_sma_slope(self):
        """ Plots first deriv of new cases across regions, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            regions = defaults.PlotRegions
            y_col = 'New Cases / 100K'
            for region in regions:
                _df = self.covid_df[region]
                _df['slope'] = np.gradient(sma(_df[y_col], self.sma_win))
                _df['slope'] = sma(_df['slope'], win=7)
                ax.plot(_df.Date, _df.slope)
            ax.axhline(0, color='red', linestyle=':')
            return ax

        plot_config = {'plot_fn': _plot_fn,
                       'fname': 'new_cases_per_100K_' + str(self.sma_win) + 'd_SMA_7d_slope',
                       'title': 'New Cases 7d avg slope of ' + str(self.sma_win) + ' avg/100K',
                       'legend': defaults.PlotRegions
                       }
        self._make_plot(plot_config)

    def incidence(self):
        """ Plots two week sum of new cases, scaled per 100K pop

        :return: nothing, creates plot
        """

        def _plot_fn(ax):
            regions = defaults.PlotRegions
            y_col = 'New Cases / 100K'
            for region in regions:
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
            regions = defaults.PlotRegions
            y_col = 'New Cases / 100K'
            for region in regions:
                _df = self.covid_df[region]
                x_dates, slope, _ = smooth_slope(_df, y_col, inc_win=14, sma_win=3, spar=0.5)
                ax.plot(x_dates, slope)
            ax.axhline(0, color='red', linestyle=':')
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

        defaults.plot_defaults()

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

    def rutherford_cases(self):
        """ Calculate something good. Just do something stupid for now

        :return: nothing, calcs stat
        """

        # New cases for Rutherford
        recent_df = self.covid_df['Rutherford']

        sma_col = 'New Cases '+str(self.sma_win)+'d avg'
        std_col = sma_col+' std'

        recent_df[sma_col] = sma(recent_df['New Cases'], self.sma_win)
        recent_df[std_col] = std_dev(recent_df[sma_col], self.sma_win)

        # Print to a html file
        html_cols = ['Date', 'Total Cases', 'New Cases', sma_col]
        file_name = 'Rutherford_Cases.html'
        recent_df.sort_values('Date', ascending=False).to_html(file_name, columns=html_cols,
                                                               index=False, float_format='%0.2f')

        recent_df = recent_df.tail(self.covid_data.settings.tail_days)
        stat = recent_df['New Cases std'].iloc[-1]

        stat = stat

        return stat

    def _make_table(self, config=None):
        """ TODO: Print some tabular information of stats

        :param config: Dict TBD
        """

        config = {} if config is None else config

        if self._debug:
            print(config)
        else:
            pass
