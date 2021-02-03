#!/bin/sh

cd /work

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


#############################
# # Get data from NY Times

# All counties in NJ
echo "Date,County,State,FIPS,Total Cases,Total Deaths" > data/csv/nytimes_nj_counties.csv
curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv | \
  awk -F, '$3=="New Jersey"' >> data/csv/nytimes_nj_counties.csv

# All US states and territories
echo "Date,State,FIPS,Total Cases,Total Deaths" > data/csv/nytimes_us_states.csv
curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv | \
  sed '1d' >> data/csv/nytimes_us_states.csv

# Nation as a whole
echo "Date,Total Cases,Total Deaths" > data/csv/nytimes_us.csv
curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv | \
  sed '1d' >> data/csv/nytimes_us.csv


#############################
## Extract regions we want from the master download files above

# Extract just New Jersey from US States
head -1 data/csv/nytimes_us_states.csv > data/csv/nytimes_nj.csv
awk -F, '$2=="New Jersey"' data/csv/nytimes_us_states.csv >> data/csv/nytimes_nj.csv
rm data/csv/nytimes_us_states.csv

# Extract just Bergen County from NJ Counties
head -1 data/csv/nytimes_nj_counties.csv > data/csv/nytimes_nj_bergen.csv
awk -F, '$2=="Bergen"' data/csv/nytimes_nj_counties.csv >>  data/csv/nytimes_nj_bergen.csv
rm data/csv/nytimes_nj_counties.csv
