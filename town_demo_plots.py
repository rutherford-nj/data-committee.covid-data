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

def get_case_data():
    KEYBASE_ROOT = os.path.join('/','Volumes')
    demo_dir = os.path.join(KEYBASE_ROOT,'Keybase','team','rutherford_nj.data_committee','covid_demographics/')
    demo_fname = '2020.csv'

    _df = pd.read_csv(os.path.join(demo_dir,demo_fname), parse_dates=['Date'])
    _df.dropna(inplace=True)
    _df.loc[_df['Age']<0,'Age'] = None

    return _df

def get_death_data():
    KEYBASE_ROOT = os.path.join('/','Volumes')
    demo_dir = os.path.join(KEYBASE_ROOT,'Keybase','team','rutherford_nj.data_committee','covid_demographics/')
    demo_fname = '2020-deaths.csv'

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
    ax1.pie(sizes, colors=colors, labels=labels, autopct=make_autopct(sizes), startangle=45,
            # wedgeprops = {"edgecolor":"k", 'linewidth': 1},
            textprops={'fontsize': 14})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title, fontsize=14)
    fname = title.replace(" - ","_").replace(" ", "_")
    plt.savefig(fname + ".svg", format="svg", dpi=400)
    plt.show()

def gender_count(df):
    return df.groupby('Gender')['Date'].count()

def bake_pies(pies):
    for title in pies:
        gender_breakdown = gender_count(pies[title])
        if len(gender_breakdown) > 0:
            if '?' in gender_count(pies[title]):
                sizes = [gender_breakdown.M, gender_breakdown.F, gender_breakdown['?']]
                labels = ['Male', 'Female','Unknown']
                colors = ['LightBlue', 'Orange', 'LightGreen']
            else:
                sizes = [gender_breakdown.M, gender_breakdown.F]
                labels = ['Male', 'Female']
                colors = ['LightBlue', 'Orange']
            pie_chart(sizes=sizes,
                      labels=labels,
                      colors=colors,
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
        plt.gca().text(x_pos[i], v[0] + 0.1, '{p:.1f}%\n{n:d}'.format(p=v[1], n=v[0]), color='black',
                 horizontalalignment="center")

    fname = title.replace(" - ","_").replace(" ", "_")
    plt.savefig(fname  + ".svg", format="svg", dpi=400)
    plt.show()

# Hardwired
os.chdir('docs/demographics')

# # ========================= CASE PLOTS
#
# # Get Case Data
# demo_df = get_case_data()
# # Break up into 18 and under, and over 18
# demo_df_minors = demo_df[demo_df.Age < 19]
# demo_df_adults = demo_df[demo_df.Age >= 19]
#
# # Gender pie charts
# pies = {"2020 COVID Cases by Gender": demo_df,
#         "2020 COVID Cases by Gender - 18 and Younger": demo_df_minors,
#         "2020 COVID Cases by Gender - Over 18": demo_df_adults}
# bake_pies(pies)
#
# # Histogram with NJ age bins
# bars = {"2020 COVID Cases by Age Group": demo_df,
#         "2020 COVID Cases by Age Group - Males": demo_df[demo_df.Gender=='M'],
#         "2020 COVID Cases by Age Group - Females": demo_df[demo_df.Gender=='F']}
# for bar in bars:
#     hist_chart(df=bars[bar],
#                age_bins=[0, 4, 17, 29, 49, 64, 79, 200],
#                labels=['0-4', '5-17', '18-29', '30-49', '50-64', '65-79', '80+'],
#                title=bar)

# ========================= DEATH PLOTS

# Get Case Data
demo_df = get_death_data()
# Break up into 18 and under, and over 18
demo_df_minors = demo_df[demo_df.Age < 19]
demo_df_adults = demo_df[demo_df.Age >= 19]

# Gender pie charts
pies = {"2020 COVID Deaths by Gender": demo_df,
        "2020 COVID Deaths by Gender - 18 and Younger": demo_df_minors,
        "2020 COVID Deaths by Gender - Over 18": demo_df_adults}
# bake_pies(pies)

# Histogram with NJ age bins
bars = {"2020 COVID Deaths by Age Group": demo_df,
        "2020 COVID Deaths by Age Group - Males": demo_df[demo_df.Gender=='M'],
        "2020 COVID Deaths by Age Group - Females": demo_df[demo_df.Gender=='F']}
for bar in bars:
    hist_chart(df=bars[bar],
               age_bins=[0, 4, 17, 29, 49, 64, 79, 200],
               labels=['0-4', '5-17', '18-29', '30-49', '50-64', '65-79', '80+'],
               title=bar)

# #############################
# fig = plt.figure(figsize=(13, 9))
# ax = fig.add_subplot(1, 1, 1)
#
# # age_config = {'minors': {'data': demo_df_minors, 'legend': '18 and under', 'plt_opts': }}
#
# for df in [demo_df_minors,demo_df_adults]:
#     _df = pd.DataFrame()
#     rng = pd.date_range('2020-01-01', '2020-12-31', freq='D')
#     _df = pd.DataFrame(index=rng)
#
#     _df['New Cases'] = df.groupby('Date').Age.count()
#     _df = _df.fillna(0)
#
#     _df['sma'] = _df['New Cases'].rolling(14, min_periods=0).mean()
#     ax.plot(_df.index, _df['sma'])
#     ax.stem(_df.index, _df['New Cases'], linefmt='x:', markerfmt=' ', basefmt=' ')
#
# ax.set_ylabel('# of cases', fontsize=20)
# my_fmt = mdates.DateFormatter('%b')
# ax.xaxis.set_major_formatter(my_fmt)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# 
# plt.title('Cases by age over 2020', fontsize=24)
# plt.legend(['minors','adults','minors','adults'], fontsize=16)
# plt.grid()
#
# plt.show()
# #
# # all_mean = demo_df.mean()
# # minor_mean = demo_df.mean()
# # adult_mean = demo_df.mean()
# #
# # all_std = demo_df.std()
# # minor_std = demo_df.std()
# # adult_std = demo_df.std()

