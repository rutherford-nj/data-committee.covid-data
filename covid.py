"""
Get covid data for the 07070 and display pretty plots and tables.

This is very overengineered and was done for s+g

Greg Recine <greg@gregrecine.com> Oct 18 2020
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_totals(data, region, fname, title, save_plt=False):
    """ Brain dead routine to plot total cases

    :param data: tuple of
            x_data: x-axis data (date list or array)
            y_data: y-axis data (numeric list, array or list of lists)
    :param region: area we are plotting
    :param fname: filename suffux
    :param title: chart title (str)
    :param save_plt: save to a file(T) or show to screen(F)
    :return: nada (fix this)
    """

    x_data = data[0]
    y_data = data[1]

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x_data, y_data)

    ax.set_ylabel('# of cases', fontsize=20)
    my_fmt = mdates.DateFormatter('%m/%d')
    ax.xaxis.set_major_formatter(my_fmt)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.title(region + ': ' + title, fontsize=24)
    plt.grid()

    if save_plt:
        plt.savefig(region + '_' + fname + ".svg", format="svg")
        plt.close()
    else:
        plt.show()


def plot_avgs(data, region, fname, title, save_plt=False):
    """ Plot town cases and averages

    :param data: tuple of
            x_data: data
            y_data: the new, sma, and ewma cases
    :param region: area we are plotting
    :param fname: filename suffux
    :param title: chart title (str)
    :param save_plt: save to a file(T) or show to screen(F)
    """

    x_data = data[0]
    y_data = data[1]

    y_new = y_data[0]
    y_sma = y_data[1]
    y_ewma = y_data[2]

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x_data, y_new, color='gray', marker='o', fillstyle='none', ms=2, lw=0)
    ax.plot(x_data, y_sma, 'b-')
    ax.plot(x_data, y_ewma, 'y-')

    ax.set_ylabel('# of cases', fontsize=20)
    my_fmt = mdates.DateFormatter('%m/%d')
    ax.xaxis.set_major_formatter(my_fmt)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.title(region + ': ' + title, fontsize=24)
    plt.legend(['Cases', '7d SMA', '14d EWMA'], fontsize=16)
    plt.grid()

    if save_plt:
        plt.savefig(region + '_' + fname + ".svg", format="svg")
        plt.close()
    else:
        plt.show()


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

        config = {} if config is None else config

        self.regions = {'US': 'United_States',
                        'NJ': 'New_Jersey',
                        'Counties': 'NJ_Counties',
                        'Rutherford': 'Rutherford'}

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
                       self.regions['Counties']: 932202,  # Bergen County Population
                       self.regions['Rutherford']: 18303}

        # Smoothing params
        _sma_win = 7  # sma window in days
        _ewma_spn = 14.0  # ewma span in days

        self.population = config.get('population', _population)
        self.sma_win = config.get('sma_win', _sma_win)
        self.ewma_spn = config.get('ewma_win', _ewma_spn)

    def get_data(self, data_dir=None):
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

            data_df_dict[key] = _df

        return data_df_dict

    def do_smoothing(self, covid_df, sma_win=None, ewma_spn=None, pop=None):
        """
        Do n-day SMA and m-day EWMA smoothing
        :param covid_df: input data frame (date, total cases, new cases)
        :param sma_win: int window for SMA
        :param ewma_spn: int span for EWMA
        :param pop: population (int) for /100K data generation
        :return: dataframe with new columns
        """
        sma_win = self.sma_win if sma_win is None else sma_win
        ewma_spn = self.ewma_spn if ewma_spn is None else ewma_spn
        pop = self.population if pop is None else pop

        # Do smoothing with sma and ewma (unscaled and scaled)
        covid_df[str(sma_win) + 'd avg'] = covid_df['New Cases'].rolling(sma_win).mean()
        covid_df[str(ewma_spn) + 'd ewma'] = covid_df['New Cases'].ewm(span=ewma_spn).mean()

        covid_df['New Cases / 100K'] = covid_df['New Cases'] * (1.0E5 / pop)
        covid_df[str(sma_win) + 'd avg / 100K'] = covid_df['New Cases / 100K'].rolling(sma_win).mean()
        covid_df[str(ewma_spn) + 'd ewma / 100K'] = covid_df['New Cases / 100K'].ewm(span=ewma_spn).mean()

        return covid_df


def do_plots(covid_df, region, sma_win, ewma_spn, save_plt=False):
    """
    :param region: region to plot
    :param covid_df: dataframe to plot
    :param sma_win: sma window in days
    :param ewma_spn: ewma span in days
    :param save_plt: save to a file(T) or show to screen(F)
    """
    # Plot #1 -- Total cases
    plot_totals((covid_df['Date'], covid_df['Total Cases']),
                region, 'Totals',
                'Total Confirmed COVID positive cases',
                save_plt)

    # Plot #2 -- Raw & smoothed cases
    y_data_list = [covid_df['New Cases'],
                   covid_df[str(sma_win) + 'd avg'],
                   covid_df[str(ewma_spn) + 'd ewma']]
    plot_avgs((covid_df['Date'], y_data_list),
              region, 'Confirmed',
              'Confirmed COVID positive cases',
              save_plt)

    # Plot #3 -- Raw & smoothed cases / 100K people
    y_data_list = [covid_df['New Cases / 100K'],
                   covid_df[str(sma_win) + 'd avg / 100K'],
                   covid_df[str(ewma_spn) + 'd ewma / 100K']]
    plot_avgs((covid_df['Date'], y_data_list),
              region, 'Confirmed_per100K',
              'Confirmed COVID positive cases per 100K population',
              save_plt)


def do_tables(covid_df, save_stats=False):
    """ Print some tabular information of stats

    :param covid_df:
    :param covid_df: covid data
    :param save_stats: save to a file(T) or show to screen(F)
    """

    smallest = covid_df['7d avg'].nsmallest(10, keep='first')

    if save_stats:
        pass
        # smallest.to_csv('smallest.csv', sep=',', mode='w')
    else:
        print(smallest)


def main():
    """
    Get data and output useful stuff
    """

    covid = CovidData()

    # Get data from google sheet and put into a data frame
    covid_df_dict = covid.get_data()

    # Assuming root is where this script is
    os.chdir('docs')

    for region in covid.regions:
        key = covid.regions[region]
        covid_df_dict[key] = covid.do_smoothing(covid_df_dict[key], pop=covid.population[key])

        do_plots(covid_df_dict[key], region=region, sma_win=7, ewma_spn=14.0, save_plt=True)

        do_tables(covid_df_dict[key], save_stats=True)


if __name__ == "__main__":
    main()
