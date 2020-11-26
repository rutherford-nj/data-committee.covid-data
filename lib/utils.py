"""
Utility functions for the covid data loading and processing routines

Greg Recine <greg@gregrecine.com> Nov 26 2020
"""

import rpy2.robjects as robjects
import numpy as np


def std_dev(series, win, min_periods=1):
    """ Return rolling standard deviation

    :param series: pandas series
    :param win: rolling window
    :param min_periods: min periods to use (from start)
    :return: pandas series
    """
    return series.rolling(win).mean().rolling(win, min_periods=min_periods).std()


def sma(series, win):
    """ Return rolling simple moving average

    :param series: pandas series
    :param win: rolling window
    :return: pandas series
    """
    return series.rolling(win).mean()


def ewma(series, spn):
    """ Return rolling exponentialy weighted moving average

    :param series: pandas series
    :param spn: rolling window
    :return: pandas series
    """
    return series.ewm(span=spn).mean()


def wow(cases_df):
    """ Return dataframe with day on week and week-over-week difference

    :param cases_df: pandas dataframe ['Date', value to g=calc wow'
    :return: pandas dataframe
    """
    _df = cases_df.copy()
    _df['day_of_week'] = _df['Date'].dt.weekday
    _df = _df.set_index('Date')
    _df['wow'] = _df.groupby('day_of_week').diff()
    return _df[['day_of_week', 'wow']]


def incidence(series, win):
    """ The occurrence of new cases of disease (COVID-19) in a population over a specific period of time.

    :param series: pandas series
    :param win: time period
    :return: pandas series
    """
    return series.rolling(win).sum()


def smooth_slope(cases_df, y_col):
    """ Return the slope of a regularized smooth spline of the incidence SMA

    :param cases_df:
    :param y_col:
    :return:
    """
    _df = cases_df.reset_index(drop=True)
    _df['incidence'] = incidence(_df[y_col], 14)
    _df['incidence_sma'] = sma(_df['incidence'], 3)

    x_val = _df.incidence_sma.dropna().index.tolist()
    y_val = _df.incidence_sma.dropna().values
    x_dates = _df[_df.index.isin(x_val)]['Date']

    r_x = robjects.FloatVector(x_val)
    r_y = robjects.FloatVector(y_val)

    r_smooth_spline = robjects.r['smooth.spline']  # extract R function
    spline_xy = r_smooth_spline(x=r_x, y=r_y, spar=0.5)
    spline = np.array(robjects.r['predict'](spline_xy, robjects.FloatVector(x_val)).rx2('y'))
    slope = np.gradient(spline)

    return x_dates, slope
