#!/bin/sh

cd /work

#############################
## Get data from NY Times

# All counties in NJ
echo "Date,County,State,FIPS,Total Cases,Total Deaths" > data/csv/nytimes_nj_counties.csv
curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2020.csv | \
  awk -F, '$3=="New Jersey"' >> data/csv/nytimes_nj_counties.csv
curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2021.csv | \
  awk -F, '$3=="New Jersey"' >> data/csv/nytimes_nj_counties.csv
curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2022.csv | \
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
