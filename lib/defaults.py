"""
Defaults for the covid data loading and processing routines

This is very overengineered and was done for s+g

Greg Recine <greg@gregrecine.com> Oct 30 2020
"""
import os

import seaborn as sns
from cycler import cycler

# Set some default params for data smoothing
SmoothingParams = {
    # Simple moving average window
    'SMA_WIN': int(os.environ.get('COVID_SMA_WIN'), base=10) or 14,

    # Exponentially Weighted Moving Average span
    'EWMA_SPAN': 14  # spn=29, com=14 | spn=15, com=7
}

RegionNames = {'US': 'United_States',
               'NJ': 'New_Jersey',
               'Bergen': 'Bergen_County',
               'Counties': 'NJ_Counties',
               'Rutherford': 'Rutherford'}

# The data we're importing default file names
DataFileNames = {RegionNames['US']: 'covid_tracking_us.csv',
                 RegionNames['NJ']: 'covid_tracking_nj.csv',
                 RegionNames['Counties']: 'nytimes_nj_counties.csv',
                 RegionNames['Rutherford']: 'rutherford_data.csv'}

# 2019 populations
DefaultPopulations = {RegionNames['US']: 328.2E6,
                      RegionNames['NJ']: 8.882E6,
                      RegionNames['Bergen']: 932202,
                      RegionNames['Rutherford']: 18303}
DefaultPopulations[RegionNames['Counties']] = (DefaultPopulations[RegionNames['NJ']]
                                               - DefaultPopulations[RegionNames['Bergen']])


def set_plot_defaults():
    """ Set up ploting defaults """

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
