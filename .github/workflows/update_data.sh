#!/bin/sh

set -x

# Run scraper.
cd data && go run *.go && cd ../

# Ensure rutherford data is the CSV, not an error page.
grep "Date,Total Cases,New Cases" data/csv/rutherford_data.csv || exit 0

# Setup local git config.
git config user.name github-actions
git config user.email github-actions@github.com

# Commit data changes.
timestamp=$(TZ=America/New_York date)
git commit -am "Latest data: ${timestamp}." || exit 0

# Build image for running graphing code.
docker build -t rutherford_covid_image .
docker run -v `pwd`:/work --entrypoint="/work/run.sh" rutherford_covid_image

# Run svgo to optimize SVG images.
docker run -v `pwd`:/work --entrypoint="/work/data/svgo.sh" node:15.0.1-alpine3.12

# Create timestamp file.
TZ=America/New_York date +"%A %B %d at %l:%M%P" | tee docs/last_updated

# Commit web-ready files and push.
git commit -am "Update graphs and timestamp."
git push
