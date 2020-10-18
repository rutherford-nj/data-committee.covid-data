"""
Get covid data for the 07070 and display pretty plots and tables.

This is very overengineered and was done for s+g

Greg Recine <greg@gregrecine.com> Oct 18 2020
"""
from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime
from tabulate import tabulate
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_totals(x, y, title, save_plt=False):
    """ Brain dead routine to plot total cases

    :param x: x-axis data (date list or array)
    :param y: y-axis data (numeric list, array or list of lists)
    :param title: chart title (str)
    :param save_plt: save to a file(T) or show to screen(F)
    :return: nada (fix this)
    """
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, y)

    ax.set_ylabel('# of cases', fontsize=20)
    my_fmt = mdates.DateFormatter('%m/%d')
    ax.xaxis.set_major_formatter(my_fmt)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.title(title, fontsize=24)
    plt.grid()

    if save_plt:
        plt.savefig(title + ".svg", format="svg")
        plt.close()
    else:
        plt.show()


def plot_avgs(x, y_new, y_sma, y_ewma, title, save_plt=False):
    """ Plot town cases and averages

    :param x: date
    :param y_new: numeric list of new cases
    :param y_sma: numeric list of n-day SMA
    :param y_ewma: numeric list of n-day EWMA
    :param save_plt: save to a file(T) or show to screen(F)
    :param title: chart title (str)
    """
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, y_new, color='gray', marker='o', fillstyle='none', ms=2, lw=0)
    ax.plot(x, y_sma, 'b-')
    ax.plot(x, y_ewma, 'y-')

    ax.set_ylabel('# of cases', fontsize=20)
    my_fmt = mdates.DateFormatter('%m/%d')
    ax.xaxis.set_major_formatter(my_fmt)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.title(title, fontsize=24)
    plt.legend(['Cases', '7d SMA', '14d EWMA'], fontsize=16)
    plt.grid()

    if save_plt:
        plt.savefig(title + ".svg", format="svg")
        plt.close()
    else:
        plt.show()


def date_parse(s):
    """ Convert month/day to dateimte

    :return: datetime.date
    """
    dt = datetime.strptime(s, '%m/%d/%Y')
    return dt.date()


class CovidData(object):
    """
    Class that gets and massages town covid data
    """

    def __init__(self, config=None):
        """

        :param config:
            pubhtml_id: pubhtml ID (2PACX-<ID>) of the sheet we're importing
            population:  Rutherford 2019 population
            sma_win: sma window in days
            ewma_spn: ewma span in days
            tab_num: index of the sheet tab to grab data from [DEFAULT: 0, first tab]
            header_row: index of which row contains the header [DEFAULT: 1, 2nd row since 1st is blank]
            num_cols: now many columns we are taking [DEFAULT: 3, date, total cases, new cases]
        """

        config = {} if config is None else config

        # Info on the sheet we're importing
        self.pubhtml_id = '1vS00GBGJKB0Xwtru3Rn5WrPqur19j--CibdM5R1tbnis0W_Bp18EmLFkJJc5sG4dwvMyqCorSVhHwik'
        self.tab_num = 0
        self.header_row = 1
        self.num_cols = 3

        # Smoothing params
        _population = 18460.0  # Rutherford 2019 population
        _sma_win = 7  # sma window in days
        _ewma_spn = 14.0  # ewma span in days

        self.population = config.get('population', _population)
        self.sma_win = config.get('sma_win', _sma_win)
        self.ewma_spn = config.get('ewma_win', _ewma_spn)

    def scrape_gsheet(self):
        """ Grab data from the google sheet and pop into a pandas dataframe

        :return: df: dataframe of: Date, New Cases, Total Cases
        """
        html = requests.get(
            'https://docs.google.com/spreadsheets/d/e/2PACX-' + self.pubhtml_id + '/pubhtml').text
        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")

        # Make the scraped data into a list of list
        data = []
        for table in tables:
            row = [[td.text for td in row.find_all("td")] for row in table.find_all("tr")]
            data.append(row)

        # Make the list of lists into a dataframe (yeah, I could prob do this in
        # one step above, but this works, isn't slow, and I'm lazy)
        sheet = data[self.tab_num]
        cols = sheet[self.header_row]
        vals = sheet[self.header_row + 1:]

        df = pd.DataFrame([v[:self.num_cols] for v in vals], columns=cols[:self.num_cols])
        df.Date = df.Date.apply(date_parse)
        df.set_index('Date', inplace=True)
        for c in df.columns:
            df[c] = df[c].astype(float)

        return df

    def do_smoothing(self, df, sma_win=None, ewma_spn=None, pop=None):
        """
        Do n-day SMA and m-day EWMA smoothing
        :param df: input data frame (date, total cases, new cases)
        :param sma_win: int window for SMA
        :param ewma_spn: int span for EWMA
        :param pop: population (int) for /100K data generation
        :return: dataframe with new columns
        """
        sma_win = self.sma_win if sma_win is None else sma_win
        ewma_spn = self.ewma_spn if ewma_spn is None else ewma_spn
        pop = self.population if pop is None else pop

        # Do smoothing with sma and ewma (unscaled and scaled)
        df[str(sma_win) + 'd avg'] = df['New Cases'].rolling(sma_win).mean()
        df[str(ewma_spn) + 'd ewma'] = df['New Cases'].ewm(span=ewma_spn).mean()

        df['New Cases / 100K'] = df['New Cases'] * (1.0E5 / pop)
        df[str(sma_win) + 'd avg / 100K'] = df['New Cases / 100K'].rolling(sma_win).mean()
        df[str(ewma_spn) + 'd ewma / 100K'] = df['New Cases / 100K'].ewm(span=ewma_spn).mean()

        return df


def do_plots(covid_df, sma_win, ewma_spn, save_plt=False):
    """
    :param covid_df: dataframe to plot
    :param sma_win: sma window in days
    :param ewma_spn: ewma span in days
    :param save_plt: save to a file(T) or show to screen(F)
    """
    # Plot #1 -- Total cases
    plot_totals(covid_df.index, covid_df['Total Cases'],
                'Total Confirmed COVID positive cases',
                save_plt)

    # Plot #2 -- Raw & smoothed cases
    plot_avgs(covid_df.index, covid_df['New Cases'],
              covid_df[str(sma_win) + 'd avg'],
              covid_df[str(ewma_spn) + 'd ewma'],
              'Confirmed COVID positive cases',
              save_plt)

    # Plot #3 -- Raw & smoothed cases / 100K people
    plot_avgs(covid_df.index, covid_df['New Cases / 100K'],
              covid_df[str(sma_win) + 'd avg / 100K'],
              covid_df[str(ewma_spn) + 'd ewma / 100K'],
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
        smallest.to_csv('smallest.csv', sep=',', mode='w')
    else:
        print(tabulate(covid_df.tail(7), headers='keys'))
        print(smallest)


def main():
    """
    Get data and output useful stuff
    """

    # Set director we're saving files in
    # make this a input along with sma_win, ewma_span, and save flags
    home_dir = os.environ['HOME']
    out_dir = os.path.join(home_dir, 'Working')
    os.chdir(out_dir)

    covid = CovidData()

    # Get data from google sheet and put into a data frame
    covid_df = covid.scrape_gsheet()

    covid_df = covid.do_smoothing(covid_df)

    do_plots(covid_df, sma_win=7, ewma_spn=14.0, save_plt=True)

    do_tables(covid_df, save_stats=True)


if __name__ == "__main__":
    main()
