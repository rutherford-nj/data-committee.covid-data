"""
Library to get covid demographic data for the 07070 and display pretty plots and tables.

This should be folded into the main lib workflow as an option, eventually

THIS CAN ONLY BE RUN FROM MY LAPTOP SINCE THE DATA IS IN KEYBASE

Greg Recine <greg@gregrecine.com> Jan 28 2021
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

DEMO_DIR = os.path.join('/','Users','greg','Working', 'covid_demographics/')
CASES_FILE = '2021.csv'
# DEATHS_FILE = '2020-deaths.csv'


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
    plt.savefig(fname + ".svg", format="svg", dpi=400)
    plt.show()


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
    plt.savefig(fname + ".svg", format="svg", dpi=400)
    plt.show()


def cook_bars(_bars, offset=4.0):
    age_bins = [0, 4, 17, 29, 49, 64, 79, 200]
    labels = ['0-4', '5-17', '18-29', '30-49', '50-64', '65-79', '80+']
    for bar in _bars:
        hist_chart(df=_bars[bar],
                   age_bins=age_bins,
                   labels=labels,
                   title=bar,
                   offset=offset)


# Hardwired
os.chdir('docs/demographics')

# ========================= GET DATA

# Get Case Data
demo_case_df = get_data(CASES_FILE)
# Break up into 18 and under, and over 18
demo_case_df_minors = demo_case_df[demo_case_df.Age < 19]
demo_case_df_adults = demo_case_df[demo_case_df.Age >= 19]

# # Get Death Data
# demo_death_df = get_data(DEATHS_FILE)
# # Break up into 18 and under, and over 18
# demo_death_df_minors = demo_death_df[demo_death_df.Age < 19]
# demo_death_df_adults = demo_death_df[demo_death_df.Age >= 19]

# ========================= CASE PLOTS

# Gender pie charts
pies = {"2021 COVID Cases by Gender": demo_case_df,
        "2021 COVID Cases by Gender - 18 and Younger": demo_case_df_minors,
        "2021 COVID Cases by Gender - Over 18": demo_case_df_adults}
bake_pies(pies)

# Histogram with NJ age bins
bars = {"2021 COVID Cases by Age Group": demo_case_df,
        "2021 COVID Cases by Age Group - Males": demo_case_df[demo_case_df.Gender == 'M'],
        "2021 COVID Cases by Age Group - Females": demo_case_df[demo_case_df.Gender == 'F']}
cook_bars(bars)


# ========================= DEATH PLOTS

# # Gender pie charts
# pies = {"2020 COVID Deaths by Gender": demo_death_df,
#         "2020 COVID Deaths by Gender - 18 and Younger": demo_death_df_minors,
#         "2020 COVID Deaths by Gender - Over 18": demo_death_df_adults}
# bake_pies(pies)
#
# # Histogram with NJ age bins
# bars = {"2020 COVID Deaths by Age Group": demo_death_df,
#         "2020 COVID Deaths by Age Group - Males": demo_death_df[demo_death_df.Gender == 'M'],
#         "2020 COVID Deaths by Age Group - Females": demo_death_df[demo_death_df.Gender == 'F']}
# cook_bars(bars, 0.2)
