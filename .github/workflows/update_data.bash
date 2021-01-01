#!/bin/bash

# Run scraper.
echo "::group::Fetch Data"
./data/fetch.sh
echo "::endgroup::"

# Ensure rutherford data is the CSV, not an error page.
grep "Date,Total Cases,New Cases" data/csv/rutherford_data.csv || exit 0

# Commit data changes. For scheduled runs, bail out of the workflow if there hasn't been a change in the data.
timestamp=$(TZ=America/New_York date)
if [[ "${GITHUB_EVENT_NAME}" == "schedule" ]]; then
  git commit -am "Latest data: ${timestamp}." || exit 0
else
  git commit -am "Latest data: ${timestamp}."
fi

rm docs/*

COVID_SMA_WIN=7 ./run.sh
COVID_SMA_WIN=14 ./run.sh

# Run svgo to optimize SVG images.
echo "::group::Optimize SVGs"
./data/svgo.sh
echo "::endgroup::"

# Ensure the files needed for rutherfordboronj.com are available, or exit.
[ -f docs/total_cases_per_100K_14d_SMA.svg ] && [ -f docs/new_cases_per_100K_14d_SMA.svg ] || exit 1

# Create timestamp file.
TZ=America/New_York date +"%A %B %d at %l:%M%P" | tee docs/last_updated

# Commit web-ready files and push.
git add docs
git commit -m "Update graphs and timestamp."
git push
