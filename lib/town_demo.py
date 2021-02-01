"""
Library to get covid demographic data for the 07070 and display pretty plots and tables.

This should be folded into the main lib workflow as an option, eventually

Greg Recine <greg@gregrecine.com> Jan 28 2021
"""
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEYBASE_ROOT = os.path.join('/','Volumes')

demo_dir = os.path.join(KEYBASE_ROOT,'Keybase','team','rutherford_nj.data_committee','covid_demographics/')

demo_fname = '2020.csv'

# cols = ['Gender','Age','Date']
demo_df = pd.read_csv(os.path.join(demo_dir,demo_fname), parse_dates=['Date'])
demo_df.dropna(inplace=True)
demo_df.loc[demo_df['Age']<0,'Age'] = None

# Histogram with NJ age bins
demo_df.Age.hist(bins=[0,5,17,30,50,65,80,100])
plt.show()

# Break up into 18 and under, and over 18
demo_df_minors = demo_df[demo_df.Age < 19]
demo_df_adults = demo_df[demo_df.Age >= 19]

# Histogram with NJ age bins
demo_df.Age.hist(bins=[0,5,17,30,50,65,80,100])
demo_df.Age.hist(bins=[0,5,17,30,50,65,80,100])
demo_df.Age.hist(bins=[0,5,17,30,50,65,80,100])
plt.show()

# Plot minors and adult cases over time
demo_df_minors.groupby('Date').Age.count().plot(style='+')
demo_df_adults.groupby('Date').Age.count().plot(style='x')
plt.legend(['minors','adults'])
plt.show()

all_mean = demo_df.mean()
minor_mean = demo_df.mean()
adult_mean = demo_df.mean()

all_std = demo_df.std()
minor_std = demo_df.std()
adult_std = demo_df.std()

