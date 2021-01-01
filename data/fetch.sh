#!/bin/sh

echo "Date,New Cases,Total Cases" > data/csv/covid_tracking_us.csv
curl https://api.covidtracking.com/v1/us/daily.json | \
  jq -r '.[] | {d: .date|tostring, p: .positive, pi: .positiveIncrease} |
      select(.p != null) | select(.pi != null) |
      {d: (.d[:4] + "-" + .d[4:6] + "-" + .d[6:]), p: .p, pi: .pi} |
      [.d, .pi, .p] | @csv' \
  >> data/csv/covid_tracking_us.csv

echo "Date,New Cases,Total Cases" > data/csv/covid_tracking_nj.csv
curl https://api.covidtracking.com/v1/states/nj/daily.json | \
  jq -r '.[] | {d: .date|tostring, p: .positive, pi: .positiveIncrease} |
      select(.p != null) | select(.pi != null) |
      {d: (.d[:4] + "-" + .d[4:6] + "-" + .d[6:]), p: .p, pi: .pi} |
      [.d, .pi, .p] | @csv' \
  >> data/csv/covid_tracking_nj.csv

curl https://docs.google.com/spreadsheets/d/e/2PACX-1vS00GBGJKB0Xwtru3Rn5WrPqur19j--CibdM5R1tbnis0W_Bp18EmLFkJJc5sG4dwvMyqCorSVhHwik/pub?output=csv \
  -L --output data/csv/rutherford_data.csv

echo "Date,County,State,FIPS,Total Cases,Total Deaths" > data/csv/nytimes_nj_counties.csv
curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv | \
  awk -F, '$3=="New Jersey"' >> data/csv/nytimes_nj_counties.csv
