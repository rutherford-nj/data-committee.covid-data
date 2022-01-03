"""
Library to get covid demographic data for the 07070 and display pretty plots and tables.

This should be folded into the main lib workflow as an option, eventually

THIS CAN ONLY BE RUN FROM MY LAPTOP SINCE THE DATA IS IN KEYBASE

Greg Recine <greg@gregrecine.com> Jan 28 2021

The important ENVs to set are: YEAR, DEMO_DATA, CASES_FILE

Example: YEAR='2020' CASES_FILE='2020.csv' DEATHS_FILE='2020-deaths.csv' python ./town_demo_plots.py

Env var config input:
    YEAR [str] = calendar year you're running for (used for plot title and quarterly plots)
    PLT_FMT [str] = 'jpg' or 'svg
    DPI [int] = plot dpi
    DEMO_DIR [str] = where your demo data is located
    CASES_FILE [str] = name of the csv file for cases
    DEATHS_FILE [str] = name of the csv file for deaths
    OUT_DIR [str] = output directory
    QUARTERS [bool] = Do plots for quarterly demographics
    PLOT_TO_SCREEN [bool] = Show plots on screen as processing
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

config = {
    'YEAR': '2021',
    'PLT_FMT': 'svg',
    'DPI': 400,
    'DEMO_DIR': os.path.join('/','Users','greg','Working', 'covid_demographics/'),
    'CASES_FILE': None,
    'DEATHS_FILE': None,
    'OUT_DIR': os.path.join(os.getcwd(),'docs','demographics'),
    'QUARTERS': False,
    'PLOT_TO_SCREEN': False,
}

YEAR = os.environ.get('YEAR', config['YEAR'])
PLT_FMT = os.environ.get('PLT_FMT', config['PLT_FMT'])
DPI = os.environ.get('DPI', config['DPI'])
DEMO_DIR = os.environ.get('DEMO_DIR', config['DEMO_DIR'])
CASES_FILE = os.environ.get('CASES_FILE', config['CASES_FILE'])
DEATHS_FILE = os.environ.get('DEATHS_FILE', config['DEATHS_FILE'])
OUT_DIR = os.environ.get('OUT_DIR', config['OUT_DIR'])
QUARTERS = os.environ.get('QUARTERS', config['QUARTERS'])
PLOT_TO_SCREEN = os.environ.get('PLOT_TO_SCREEN', config['PLOT_TO_SCREEN'])

def get_data(demo_fname):

    _df = pd.read_csv(os.path.join(DEMO_DIR, demo_fname), parse_dates=['Date'])
    _df.dropna(inplace=True)
    _df.loc[_df['Age'] < 0, 'Age'] = None

    return _df


# Fn for pie chart % and value in wedge
def make_autopct(sizes):
    def my_autopct(pct):
        total = sum(sizes)
        val = int(round(pct * total / 100.0))
        return '{p:.1f}%\n{v:d}'.format(p=pct, v=val)

    return my_autopct


# DAMN PIE CHARTS
def pie_chart(sizes, labels, colors, title=None):
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, colors=colors, labels=labels, autopct=make_autopct(sizes), startangle=45,
            # wedgeprops = {"edgecolor":"k", 'linewidth': 1},
            textprops={'fontsize': 14})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title, fontsize=14)
    fname = title.replace(" - ", "_").replace(" ", "_")
    plt.savefig(fname + "."+PLT_FMT, format=PLT_FMT, dpi=DPI)
    if PLOT_TO_SCREEN:
        plt.show()
    plt.close()


def gender_count(df):
    return df.groupby('Gender')['Date'].count()


def bake_pies(_pies):
    for title in _pies:
        gender_breakdown = gender_count(_pies[title])
        if len(gender_breakdown) > 0:
            if '?' in gender_count(_pies[title]):
                sizes = [gender_breakdown.M, gender_breakdown.F, gender_breakdown['?']]
                labels = ['Male', 'Female', 'Unknown']
                colors = ['LightBlue', 'Orange', 'LightGreen']
            else:
                sizes = [gender_breakdown.M, gender_breakdown.F]
                labels = ['Male', 'Female']
                colors = ['LightBlue', 'Orange']
            pie_chart(sizes=sizes,
                      labels=labels,
                      colors=colors,
                      title=title)


def hist_chart(df, age_bins, labels, title, offset):
    hist_df = pd.DataFrame()
    hist_df['vals'] = df.Age.value_counts(bins=age_bins, sort=False)
    hist_df['pct'] = 100.0 * df.Age.value_counts(bins=age_bins, sort=False, normalize=True)
    hist_df['label'] = labels

    x_pos = [i for i, _ in enumerate(hist_df.label.to_list())]

    plt.bar(x_pos, hist_df.vals.to_list(), color='slategrey', edgecolor='black', clip_on=False)
    plt.xticks(x_pos, labels)
    plt.yticks([])

    plt.title(title, fontsize=16, pad=20)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tick_params(axis='x', which='both', bottom=False, top=False)

    for i, v in enumerate(list(zip(hist_df.vals, hist_df.pct))):
        plt.gca().text(x_pos[i], v[0] + offset, '{p:.1f}%\n{n:d}'.format(p=v[1], n=v[0]), color='black',
                       horizontalalignment="center")

    fname = title.replace(" - ", "_").replace(" ", "_")
    plt.savefig(fname + "."+PLT_FMT, format=PLT_FMT, dpi=DPI)
    if PLOT_TO_SCREEN:
        plt.show()
    plt.close()


def cook_bars(_bars, offset=4.0):
    age_bins = [0, 4, 17, 29, 49, 64, 79, 200]
    labels = ['0-4', '5-17', '18-29', '30-49', '50-64', '65-79', '80+']
    for bar in _bars:
        hist_chart(df=_bars[bar],
                   age_bins=age_bins,
                   labels=labels,
                   title=bar,
                   offset=offset)


def make_charts(df, flavor, bar_offset=4.0, age=19):
    # Gender pie charts
    pies = {YEAR+" COVID "+flavor+" by Gender": df,
            YEAR+" COVID "+flavor+" by Gender - 18 and Younger": df[df['Age'] < age],
            YEAR+" COVID "+flavor+" by Gender - Over 18": df[df['Age'] >= age]}
    bake_pies(pies)

    # Histogram with NJ age bins
    bars = {YEAR+" COVID "+flavor+" by Age Group": df,
            YEAR+" COVID "+flavor+" by Age Group - Males": df[df['Gender'] == 'M'],
            YEAR+" COVID "+flavor+" by Age Group - Females": df[df['Gender'] == 'F']}
    cook_bars(bars, bar_offset)


def quarterly_plots(df, year):
    # Quarterly Plots
    quarters = {
        'Q1': (pd.Timestamp(year, 1, 1), pd.Timestamp(year, 4, 1)),
        'Q2': (pd.Timestamp(year, 4, 1), pd.Timestamp(year, 7, 1)),
        'Q3': (pd.Timestamp(year, 7, 1), pd.Timestamp(year, 10, 1)),
        'Q4': (pd.Timestamp(year, 10, 1), pd.Timestamp(year+1, 1, 1)),
    }
    bar_title = str(year)+" COVID Cases by Age Group "
    bars = {
        bar_title + q: df[(df['Date']>=quarters[q][0]) & (df['Date']<quarters[q][1])]
        for q in quarters
    }
    cook_bars(bars)


def main():
    # Hardwired -- fix this
    os.chdir(OUT_DIR)

    # CASE PLOTS
    if CASES_FILE is not None:
        case_df = get_data(CASES_FILE)

        make_charts(case_df, flavor='Cases')
        
        if QUARTERS:
            quarterly_plots(case_df, int(YEAR))

    #DEATH PLOTS
    if DEATHS_FILE is not None:
        make_charts(get_data(DEATHS_FILE), flavor='Deaths', bar_offset=0.2)

if __name__ == "__main__":
    main()