#!/bin/sh

cd /work

curl https://docs.google.com/spreadsheets/d/e/2PACX-1vS00GBGJKB0Xwtru3Rn5WrPqur19j--CibdM5R1tbnis0W_Bp18EmLFkJJc5sG4dwvMyqCorSVhHwik/pub?output=csv \
  -L --output data/csv/rutherford_data.csv

echo "Date,County,State,FIPS,Total Cases,Total Deaths" > data/csv/nytimes_nj_counties.csv
curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv | \
  awk -F, '$3=="New Jersey"' >> data/csv/nytimes_nj_counties.csv

curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv \
  -L --output data/csv/nytimes_us_states.csv

curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv \
  -L --output data/csv/nytimes_us.csv
