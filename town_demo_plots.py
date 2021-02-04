"""
Library to get covid demographic data for the 07070 and display pretty plots and tables.

This should be folded into the main lib workflow as an option, eventually

THIS CAN ONLY BE RUN FROM MY LAPTOP SINCE THE DATA IS IN KEYBASE

Greg Recine <greg@gregrecine.com> Jan 28 2021
"""
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lib.defaults as defaults

def get_data():
    KEYBASE_ROOT = os.path.join('/','Volumes')
    demo_dir = os.path.join(KEYBASE_ROOT,'Keybase','team','rutherford_nj.data_committee','covid_demographics/')
    demo_fname = '2020.csv'

    _df = pd.read_csv(os.path.join(demo_dir,demo_fname), parse_dates=['Date'])
    _df.dropna(inplace=True)
    _df.loc[_df['Age']<0,'Age'] = None

    return _df

# Fn for pie chart % and value in wedge
def make_autopct(sizes):
    def my_autopct(pct):
        total = sum(sizes)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%\n{v:d}'.format(p=pct,v=val)
    return my_autopct

# DAMN PIE CHARTS
def pie_chart(sizes, labels, colors, title=None):
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, colors=colors, labels=labels, autopct=make_autopct(sizes), startangle=90,
            wedgeprops = {"edgecolor":"k", 'linewidth': 3},
            textprops={'fontsize': 14})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title, fontsize=14)
    plt.savefig(title.replace(" ", "_") + ".svg", format="svg", dpi=400)
    plt.show()

def gender_count(df):
    return df.groupby('Gender')['Date'].count()

def bake_pies(pies):
    for title in pies:
        pie_chart(sizes=[gender_count(pies[title]).M, gender_count(pies[title]).F],
                  labels=['Male', 'Female'],
                  colors=['LightBlue','Orange'],
                  title=title)

def hist_chart(df, age_bins, labels, title):
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
        plt.gca().text(x_pos[i], v[0] + 4.0, '{p:.1f}%\n{n:d}'.format(p=v[1], n=v[0]), color='black',
                 horizontalalignment="center")

    plt.savefig(title.replace(" ", "_") + ".svg", format="svg", dpi=400)
    plt.show()

# =========================

# Get Data
demo_df = get_data()
# Break up into 18 and under, and over 18
demo_df_minors = demo_df[demo_df.Age < 19]
demo_df_adults = demo_df[demo_df.Age >= 19]

# Hardwired
os.chdir('docs/demographics')

# Gender pie charts
pies = {"2020 COVID Cases by Gender": demo_df,
        "2020 COVID Cases by Gender - 18 and Younger": demo_df_minors,
        "2020 COVID Cases by Gender - Over 18": demo_df_adults}
bake_pies(pies)

# Histogram with NJ age bins
bars = {"2020 COVID Cases by Age Group": demo_df,
        "2020 COVID Cases by Age Group - Males": demo_df[demo_df.Gender=='M'],
        "2020 COVID Cases by Age Group - Females": demo_df[demo_df.Gender=='F']}
for bar in bars:
    hist_chart(df=bars[bar],
               age_bins=[0, 4, 17, 29, 49, 64, 79, 200],
               labels=['0-4', '5-17', '18-29', '30-49', '50-64', '65-79', '80+'],
               title=bar)

#
# all_mean = demo_df.mean()
# minor_mean = demo_df.mean()
# adult_mean = demo_df.mean()
#
# all_std = demo_df.std()
# minor_std = demo_df.std()
# adult_std = demo_df.std()

