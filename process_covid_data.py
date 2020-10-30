"""
Get covid data for the 07070 and display pretty plots and tables.

This is very overengineered and was done for s+g

Greg Recine <greg@gregrecine.com> Oct 18 2020
"""
import os

from lib.defaults import SmoothingParams, MakeThesePlots
from lib import covid


def main():
    """
    Get data and output useful stuff
    """

    # Init covid class
    # TODO: Make an argument
    sma_win = SmoothingParams['SMA_WIN']
    ewma_span = SmoothingParams['EWMA_SPAN']
    settings = {
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

    # Set up the run
    covid_settings = covid.Settings(settings)

    # Get the data
    covid_data = covid.GetData(covid_settings)

    # Dir to write output to (assuming root is where this script is)
    os.chdir('docs')

    # Set up plotting and plot away if set
    plot_data = covid.MakePlots(covid_data)

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
