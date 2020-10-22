"""
Get covid data for the 07070 and display pretty plots and tables.

This is very overengineered and was done for s+g

Greg Recine <greg@gregrecine.com> Oct 18 2020
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# pylint: disable=R0902
# I have 8/7 instance attributes, need to refactor?
class CovidData:
    """
    Class that gets and massages town covid data
    """

    def __init__(self, config=None):
        """

        :param config:
            data_files: dict of file description and file name
            population:  Rutherford 2019 population
            sma_win: sma window in days
            ewma_spn: ewma span in days
        """

        # Set config to empty dict if not provided
        config = {} if config is None else config

        # Set regions
        self.regions = {'US': 'United_States',
                        'NJ': 'New_Jersey',
                        'Bergen': 'Bergen_County',
                        'Counties': 'NJ_Counties',
                        'Rutherford': 'Rutherford'}

        # Smoothing defaults
        _sma_win = 7
        _ewma_spn = 14.0
        self.sma_win = config.get('sma_win', _sma_win)
        self.ewma_spn = config.get('ewma_spn', _ewma_spn)

        # The data we're importing
        _data_files = {self.regions['US']: 'covid_tracking_us.csv',
                       self.regions['NJ']: 'covid_tracking_nj.csv',
                       self.regions['Counties']: 'nytimes_nj_counties.csv',
                       self.regions['Rutherford']: 'rutherford_data.csv'}

        self.data_files = config.get('data_files', _data_files)

        _data_dir = os.path.join(os.getcwd(), 'data', 'csv')
        self.data_dir = config.get('data_dir', _data_dir)

        # 2019 populations
        _population = {self.regions['US']: 328.2E6,
                       self.regions['NJ']: 8.882E6,
                       self.regions['Bergen']: 932202,  # Bergen County Population
                       self.regions['Rutherford']: 18303}

        _population[self.regions['Counties']] = _population[self.regions['NJ']] - _population[self.regions['Bergen']]

        self.population = config.get('population', _population)

        # Get data from sources
        self.data_df_dict = self._get_data()

        # Break out counties
        # self.data_df_dict = self._breakout()

        # Do smoothing
        for key in self.regions:
            region = self.regions[key]
            self.data_df_dict[region] = self.do_smoothing(self.data_df_dict[region], region)

    def _get_data(self, data_dir=None):
        """ Grab data from the google sheet and pop into a pandas dataframe

        :return: df: dataframe of: Date, New Cases, Total Cases
        """

        data_dir = self.data_dir if data_dir is None else data_dir

        data_df_dict = {}
        for key in self.data_files.keys():
            _df = pd.read_csv(os.path.join(data_dir, self.data_files[key]), parse_dates=[0])
            if key == self.regions['Counties']:
                _df['New Cases'] = _df.groupby('County').apply(lambda x: x['Total Cases'].diff()).reset_index(level=0,
                                                                                                              drop=True)
                _df.loc[_df['New Cases'].isna(), 'New Cases'] = _df['Total Cases']

                # Break out Bergen County
                data_df_dict[self.regions['Bergen']] = _df[_df.County == 'Bergen']

                # NJ ex Bergen County
                _df = _df[_df.County != 'Bergen'].groupby('Date').sum()
                _df.reset_index(inplace=True)

            data_df_dict[key] = _df.sort_values('Date')

        return data_df_dict

    def do_smoothing(self, covid_df, region):
        """
        Do n-day SMA and m-day EWMA smoothing
        :param region: data region
        :param covid_df: input data frame (date, total cases, new cases)
        :return: dataframe with new columns
        """
        sma_win = self.sma_win
        ewma_spn = self.ewma_spn
        population = self.population[region]

        # Do smoothing with sma and ewma (unscaled and scaled)
        covid_df[str(sma_win) + 'd avg'] = covid_df['New Cases'].rolling(sma_win).mean()
        covid_df[str(ewma_spn) + 'd ewma'] = covid_df['New Cases'].ewm(span=ewma_spn).mean()

        covid_df['New Cases / 100K'] = covid_df['New Cases'] * (1.0E5 / population)
        covid_df[str(sma_win) + 'd avg / 100K'] = covid_df['New Cases / 100K'].rolling(sma_win).mean()
        covid_df[str(ewma_spn) + 'd ewma / 100K'] = covid_df['New Cases / 100K'].ewm(span=ewma_spn).mean()

        covid_df['Total Cases / 100K'] = covid_df['Total Cases'] * (1.0E5 / population)
        covid_df[str(sma_win) + 'd avg Total Cases / 100K'] = covid_df['Total Cases / 100K'].rolling(sma_win).mean()
        covid_df[str(ewma_spn) + 'd ewma Total Cases/ 100K'] = covid_df['Total Cases / 100K'].ewm(span=ewma_spn).mean()

        return covid_df

    def do_plots(self, key, config):
        """
        :param key: region to plot
        :param config [dict]
                total:
                averages:
                raw:
                scaled:
                debug: save to a file(F) or show to screen(T)
        """

        # do_total = config.get('total', True)
        # do_avgs = config.get('averages', True)
        # do_raw = config.get('raw', True)
        # do_scaled = config.get('scaled', True)
        self.debug = config.get('debug', True)

        # region = self.regions[key]

        sma_win = self.sma_win
        ewma_spn = self.ewma_spn

        # All New Cases Scaled by SMA
        def all_cases_scaled_sma(ax):
            us_df = self.data_df_dict[self.regions['US']]
            nj_df = self.data_df_dict[self.regions['NJ']]
            ct_df = self.data_df_dict[self.regions['Counties']]
            bc_df = self.data_df_dict[self.regions['Bergen']]
            rf_df = self.data_df_dict[self.regions['Rutherford']]

            ax.plot(us_df['Date'], us_df[str(sma_win) + 'd avg / 100K'])
            ax.plot(nj_df['Date'], nj_df[str(sma_win) + 'd avg / 100K'])
            ax.plot(ct_df['Date'], ct_df[str(sma_win) + 'd avg / 100K'])
            ax.plot(bc_df['Date'], bc_df[str(sma_win) + 'd avg / 100K'])
            ax.plot(rf_df['Date'], rf_df[str(sma_win) + 'd avg / 100K'])

            return ax

        plot_config = {'plot_fn': all_cases_scaled_sma,
                       'fname': 'Totals_' + str(sma_win) + 'D_SMA_per_100K',
                       'title': str(sma_win) + ' day average / 100K residents',
                       'legend': ['United States',
                                  'New Jersey',
                                  'Other Counties',
                                  'Bergen County',
                                  'Rutherford']
                       }
        self.make_plot(plot_config)

        # All New Cases Scaled by EWMA
        def all_cases_scaled_ewma(ax):
            us_df = self.data_df_dict[self.regions['US']]
            nj_df = self.data_df_dict[self.regions['NJ']]
            ct_df = self.data_df_dict[self.regions['Counties']]
            bc_df = self.data_df_dict[self.regions['Bergen']]
            rf_df = self.data_df_dict[self.regions['Rutherford']]

            ax.plot(us_df['Date'], us_df[str(ewma_spn) + 'd ewma / 100K'])
            ax.plot(nj_df['Date'], nj_df[str(ewma_spn) + 'd ewma / 100K'])
            ax.plot(ct_df['Date'], ct_df[str(ewma_spn) + 'd ewma / 100K'])
            ax.plot(bc_df['Date'], bc_df[str(ewma_spn) + 'd ewma / 100K'])
            ax.plot(rf_df['Date'], rf_df[str(ewma_spn) + 'd ewma / 100K'])

            return ax

        plot_config = {'plot_fn': all_cases_scaled_ewma,
                       'fname': 'Totals_' + str(ewma_spn) + 'D_EWMA_per_100K',
                       'title': str(ewma_spn) + ' day weighted average / 100K residents',
                       'legend': ['United States',
                                  'New Jersey',
                                  'Other Counties',
                                  'Bergen County',
                                  'Rutherford']
                       }
        self.make_plot(plot_config)

        # Total New Cases Scaled
        def total_cases_scaled(ax):
            us_df = self.data_df_dict[self.regions['US']]
            nj_df = self.data_df_dict[self.regions['NJ']]
            ct_df = self.data_df_dict[self.regions['Counties']]
            bc_df = self.data_df_dict[self.regions['Bergen']]
            rf_df = self.data_df_dict[self.regions['Rutherford']]

            ax.plot(us_df['Date'], us_df['Total Cases / 100K'])
            ax.plot(nj_df['Date'], nj_df['Total Cases / 100K'])
            ax.plot(ct_df['Date'], ct_df['Total Cases / 100K'])
            ax.plot(bc_df['Date'], bc_df['Total Cases / 100K'])
            ax.plot(rf_df['Date'], rf_df['Total Cases / 100K'])

            return ax
        plot_config = {'plot_fn': total_cases_scaled,
                       'fname': 'Totals_cases_per_100K',
                       'title': 'Total cases / 100K residents',
                       'legend': ['United States',
                                  'New Jersey',
                                  'Other Counties',
                                  'Bergen County',
                                  'Rutherford']
                       }
        self.make_plot(plot_config)

        # Total New Cases Scaled by SMA
        def total_cases_scaled_sma(ax):
            us_df = self.data_df_dict[self.regions['US']]
            nj_df = self.data_df_dict[self.regions['NJ']]
            ct_df = self.data_df_dict[self.regions['Counties']]
            bc_df = self.data_df_dict[self.regions['Bergen']]
            rf_df = self.data_df_dict[self.regions['Rutherford']]

            ax.plot(us_df['Date'], us_df[str(sma_win) + 'd avg Total Cases / 100K'])
            ax.plot(nj_df['Date'], nj_df[str(sma_win) + 'd avg Total Cases / 100K'])
            ax.plot(ct_df['Date'], ct_df[str(sma_win) + 'd avg Total Cases / 100K'])
            ax.plot(bc_df['Date'], bc_df[str(sma_win) + 'd avg Total Cases / 100K'])
            ax.plot(rf_df['Date'], rf_df[str(sma_win) + 'd avg Total Cases / 100K'])

            return ax
        plot_config = {'plot_fn': total_cases_scaled_sma,
                       'fname': 'Totals_cases_per_100K_SMA',
                       'title': 'Total cases / 100K residents -- ' + str(sma_win) + ' day average',
                       'legend': ['United States',
                                  'New Jersey',
                                  'Other Counties',
                                  'Bergen County',
                                  'Rutherford']
                       }
        self.make_plot(plot_config)

        #
        # # Plot #1 -- Total cases
        # if do_total:
        #     if do_raw:
        #         plot_config = {'x_col': 'Date',
        #                        'y_cols': ['Total Cases'],
        #                        'plot_args': [{}],
        #                        'fname': 'Totals',
        #                        'title': 'Total Confirmed COVID positive cases',
        #                        'region': region,
        #                        'legend': False
        #                        }
        #         self.make_plot(plot_config, save_plt)
        #
        # # Plot -- Raw & smoothed cases
        # if do_avgs:
        #     plot_config = {'x_col': 'Date',
        #                    'plot_args': [{'color': 'gray', 'marker': 'o', 'fillstyle': 'none', 'ms': 2, 'lw': 0},
        #                                  {'color': 'blue'},
        #                                  {'color': 'orange'}],
        #                    'region': region,
        #                    'legend': True
        #                    }
        #     if do_raw:
        #         plot_config['y_cols'] = ['New Cases', str(sma_win) + 'd avg', str(ewma_spn) + 'd ewma']
        #         plot_config['fname'] = 'Confirmed'
        #         plot_config['title'] = 'Confirmed COVID positive cases'
        #         self.make_plot(plot_config, save_plt)
        #
        #     if do_scaled:
        #         # Plot -- Raw & smoothed cases / 100K people
        #         plot_config['y_cols'] = ['New Cases / 100K', str(sma_win) + 'd avg / 100K',
        #                                  str(ewma_spn) + 'd ewma / 100K']
        #         plot_config['fname'] = 'Confirmed_per100K'
        #         plot_config['title'] = 'Confirmed COVID positive cases per 100K population'
        #         self.make_plot(plot_config, save_plt)

    def do_tables(self, key, config):
        """ Print some tabular information of stats

        :param key: region to plot
        :param config: save to a file(T) or show to screen(F)
        """
        debug = config.get('debug', True)

        region = self.regions[key]

        covid_df = self.data_df_dict[region]

        smallest = covid_df['7d avg'].nsmallest(10, keep='first')

        if self.debug:
            print(smallest)
        else:
            pass
            # smallest.to_csv('smallest.csv', sep=',', mode='w')

    def make_plot(self, config):
        """ Plot town cases and averages

        :param config: dict of
                - x_col: x columns (str)
                - y_cols: list of y columns
                - plot_args: list of args for plot
                - region: area we are plotting
                - fname: filename suffux
                - title: chart title (str)
        :param save_plt: save to a file(T) or show to screen(F)
        """

        fig = plt.figure(figsize=(13, 9))
        ax = fig.add_subplot(1, 1, 1)

        config['plot_fn'](ax)

        ax.set_ylabel('# of cases', fontsize=20)
        my_fmt = mdates.DateFormatter('%m/%d')
        ax.xaxis.set_major_formatter(my_fmt)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.title(config['title'], fontsize=24)
        if config.get('legend', None) is not None:
            plt.legend(config['legend'], fontsize=16)
        plt.grid()

        if self.debug:
            plt.show()
        else:
            plt.savefig(config['fname'] + ".svg", format="svg")
            plt.close()


def main():
    """
    Get data and output useful stuff
    """

    # Init covid class
    sma_win = 14
    ewma_span = 14 # spn=29, com=14 | spn=15, com=7
    data_config = {'sma_win': sma_win,
                   'ewma_spn': ewma_span}

    covid_data = CovidData(data_config)

    # Dir to write output to (assuming root is where this script is)
    os.chdir('docs')

    covid_data.do_plots(key=None, config={'debug': False})

    # plot_config = {'raw': True,
    #                'scaled': True,
    #                'debug': False}
    # for region in covid_data.regions:
    #     covid_data.do_plots(key=region, config=plot_config)
    #     covid_data.do_tables(key=region, config=plot_config)


if __name__ == "__main__":
    main()
