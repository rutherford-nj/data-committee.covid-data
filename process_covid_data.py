"""
Get covid data for the 07070 and display pretty plots and tables.

This is very overengineered and was done for s+g

Greg Recine <greg@gregrecine.com> Oct 18 2020
"""
import os

from lib import covid

MakeThesePlots = [
    covid.MakePlots.new_cases_norm_sma,        # All New Cases Scaled by SMA
    # covid.MakePlots.new_cases_norm_ewma,       # All New Cases Scaled by EWMA
    # covid.MakePlots.total_cases_norm,          # Total New Cases Scaled
    covid.MakePlots.total_cases_norm_sma,      # Total New Cases Scaled by SMA
    covid.MakePlots.rutherford_new_cases,      # Rutherford new cases, unscaled
    covid.MakePlots.rutherford_new_cases_reg,      # Rutherford new cases, unscaled & fitted
    covid.MakePlots.rutherford_total_cases,    # Rutherford total cases, unscaled
    covid.MakePlots.new_cases_norm_sma_wow,    # Rutherford new week-over-week cases, unscaled
    covid.MakePlots.new_cases_norm_sma_slope,  # Rutherford X day slope of scaled, SMA new cases
    covid.MakePlots.incidence,                 # Total New Cases scaled 14d sum ("Incidence")
    covid.MakePlots.trajectory,                # sma incidence slope
]

CalcTheseStats = [
    covid.MakeStats.rutherford_cases,      # Some stat type goes here
]


def main():
    """
    Get data and output useful stuff
    """

    # Set up the run with settings in lib/defaults.py
    covid_settings = covid.Settings()

    # Initialize the data
    covid_data = covid.GetData(covid_settings)

    # Dir to write output to (assuming root is where this script is)
    os.chdir('docs')

    # Set up plotting and plot away if set
    plot_data = covid.MakePlots(covid_data)

    # Plot the plots we want
    for plot in MakeThesePlots:
        plot_type = plot.__name__
        getattr(plot_data, plot_type)()

    # Set up stats and run if set
    stats_data = covid.MakeStats(covid_data)

    # Calc the stats we want
    for calc in CalcTheseStats:
        calc_type = calc.__name__
        getattr(stats_data, calc_type)()



if __name__ == "__main__":
    main()
