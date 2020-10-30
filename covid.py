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


# I have 8/7 instance attributes, need to refactor?
class CovidData:
    """
    Class that gets and massages town covid data
    """

    _regions = {'US': 'United_States',
                'NJ': 'New_Jersey',
                'Bergen': 'Bergen_County',
                'Counties': 'NJ_Counties',
                'Rutherford': 'Rutherford'}

    @property
    def regions(self):
        """ return regions dict """
        return self._regions

    @property
    def debug(self):
        """ return debug bool"""
        return self.config.get('debug', True)

    def __init__(self, config=None):
        """

        :param config:
            data_files: dict of file description and file name
            population:  Rutherford 2019 population
            sma_win: sma window in days
            ewma_spn: ewma span in days
        """

        # Set config to empty dict if not provided
        self.config = {} if config is None else config

        # Set smoothing defaults if not in config
        self.sma_win = config.get('sma_win', SmoothingParams['SMA_WIN'])
        self.ewma_spn = config.get('ewma_spn', SmoothingParams['EWMA_SPAN'])

        regions = self.regions

        # The data we're importing
        _data_files = {regions['US']: 'covid_tracking_us.csv',
                       regions['NJ']: 'covid_tracking_nj.csv',
                       regions['Counties']: 'nytimes_nj_counties.csv',
                       regions['Rutherford']: 'rutherford_data.csv'}

        self.data_files = config.get('data_files', _data_files)

        _data_dir = os.path.join(os.getcwd(), 'data', 'csv')
        self.data_dir = config.get('data_dir', _data_dir)

        # 2019 populations
        _population = {regions['US']: 328.2E6,
                       regions['NJ']: 8.882E6,
                       regions['Bergen']: 932202,  # Bergen County Population
                       regions['Rutherford']: 18303}

        _population[regions['Counties']] = _population[regions['NJ']] - _population[regions['Bergen']]

        self.population = config.get('population', _population)

        # Get data from sources
        self.data_df_dict = self._get_data()

        # Pad data with zeros back to earliest date
        self._pad_date()

        # Do smoothing
        for key in regions:
            region = regions[key]
            self.data_df_dict[region] = self.do_smoothing(self.data_df_dict[region], region)

    def _get_data(self, data_dir=None):
        """ Grab data from the google sheet and pop into a pandas dataframe

        :return: df: dataframe of: Date, New Cases, Total Cases
        """

        data_dir = self.data_dir if data_dir is None else data_dir

        regions = self.regions

        data_df_dict = {}
        for key in self.data_files.keys():
            _df = pd.read_csv(os.path.join(data_dir, self.data_files[key]), parse_dates=[0])
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

        regions = self.regions

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
        sma_win = self.sma_win
        ewma_spn = self.ewma_spn
        population = self.population[region]

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

        self.debug = covid_data.debug
        self.sma_win = covid_data.sma_win
        self.ewma_spn = covid_data.ewma_spn

        # Set up ploting defaults
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

        regions = covid_data.regions

        self.us_df = covid_data.data_df_dict[regions['US']]
        self.nj_df = covid_data.data_df_dict[regions['NJ']]
        self.ct_df = covid_data.data_df_dict[regions['Counties']]
        self.bc_df = covid_data.data_df_dict[regions['Bergen']]
        self.rf_df = covid_data.data_df_dict[regions['Rutherford']]

    def local_new_cases(self):
        """ Plot Rutherford local case numbers and -/+ 1 std

        :return: nothing, creates plot
        """

        # New cases for Rutheford
        def _plot_fn(ax):
            _df = self.rf_df

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
            _df = self.rf_df

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
            ax.plot(self.us_df['Date'], self.us_df[str(self.sma_win) + 'd avg / 100K'])
            ax.plot(self.nj_df['Date'], self.nj_df[str(self.sma_win) + 'd avg / 100K'])
            # ax.plot(self.ct_df['Date'], self.ct_df[str(self.sma_win) + 'd avg / 100K'])
            ax.plot(self.bc_df['Date'], self.bc_df[str(self.sma_win) + 'd avg / 100K'])
            ax.plot(self.rf_df['Date'], self.rf_df[str(self.sma_win) + 'd avg / 100K'])

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
            ax.plot(self.us_df['Date'], self.us_df[str(self.ewma_spn) + 'd ewma / 100K'])
            ax.plot(self.nj_df['Date'], self.nj_df[str(self.ewma_spn) + 'd ewma / 100K'])
            # ax.plot(self.ct_df['Date'], self.ct_df[str(self.ewma_spn) + 'd ewma / 100K'])
            ax.plot(self.bc_df['Date'], self.bc_df[str(self.ewma_spn) + 'd ewma / 100K'])
            ax.plot(self.rf_df['Date'], self.rf_df[str(self.ewma_spn) + 'd ewma / 100K'])

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
            ax.plot(self.us_df['Date'], self.us_df['Total Cases / 100K'])
            ax.plot(self.nj_df['Date'], self.nj_df['Total Cases / 100K'])
            # ax.plot(self.ct_df['Date'], self.ct_df['Total Cases / 100K'])
            ax.plot(self.bc_df['Date'], self.bc_df['Total Cases / 100K'])
            ax.plot(self.rf_df['Date'], self.rf_df['Total Cases / 100K'])

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
            ax.plot(self.us_df['Date'], self.us_df[str(self.sma_win) + 'd avg Total Cases / 100K'])
            ax.plot(self.nj_df['Date'], self.nj_df[str(self.sma_win) + 'd avg Total Cases / 100K'])
            # ax.plot(ct_df['Date'], self.ct_df[str(self.sma_win) + 'd avg Total Cases / 100K'])
            ax.plot(self.bc_df['Date'], self.bc_df[str(self.sma_win) + 'd avg Total Cases / 100K'])
            ax.plot(self.rf_df['Date'], self.rf_df[str(self.sma_win) + 'd avg Total Cases / 100K'])

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

        if self.debug:
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
    # TODO: Make an argument
    sma_win = SmoothingParams['SMA_WIN']
    ewma_span = SmoothingParams['EWMA_SPAN']
    data_config = {
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

    # Init class
    covid_data = CovidData(data_config)

    # Dir to write output to (assuming root is where this script is)
    os.chdir('docs')

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
