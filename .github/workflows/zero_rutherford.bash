#!/bin/bash

currenttime=$(TZ=America/New_York date +%H:%M)
if [[ "$currenttime" > "21:00" ]]; then
  today=$(TZ=America/New_York date +%-m/%-d/%Y)
  prev_cases=$(tail -n 1 ./data/csv/rutherford_data.csv | cut -d"," -f2)
  if grep -q $today ./data/csv/rutherford_data.csv; then
    echo "today's data already recorded"
  else
    echo "no cases reported today, recording 0"
    echo $today,$prev_cases,0 >> ./data/csv/rutherford_data.csv
  fi
else
  echo "still too early in the day to assume no cases"
fi