#!/bin/bash

# Build image for running code, uncomment for local image building.
# docker build -t rutherford_covid_image .

# Run scraper.

echo "::group::Fetch Data"
docker run -v `pwd`:/work --entrypoint="/work/data/fetch.sh" ghcr.io/bogosj/rutherford_covid_image
echo "::endgroup::"

# Ensure rutherford data is the CSV, not an error page.
grep "Date,Total Cases,New Cases" data/csv/rutherford_data.csv || exit 0

# Setup local git config.
git config --global user.name github-actions
git config --global user.email github-actions@github.com

# Commit data changes. For scheduled runs, bail out of the workflow if there hasn't been a change in the data.
timestamp=$(TZ=America/New_York date)
if [[ "${GITHUB_EVENT_NAME}" == "schedule" ]]; then
  git commit -am "Latest data: ${timestamp}." || exit 0
else
  git commit -am "Latest data: ${timestamp}."
fi

docker run -v `pwd`:/work --entrypoint="/work/run.sh" ghcr.io/bogosj/rutherford_covid_image
docker run -v `pwd`:/work --entrypoint="/work/run.sh" --env COVID_SMA_WIN=7 ghcr.io/bogosj/rutherford_covid_image

# Run svgo to optimize SVG images.
echo "::group::Optimize SVGs"
docker run -v `pwd`:/work --entrypoint="/work/data/svgo.sh" node:15.0.1-alpine3.12
echo "::endgroup::"

# Create timestamp file.
TZ=America/New_York date +"%A %B %d at %l:%M%P" | tee docs/last_updated

# Commit web-ready files and push.
git add docs
git commit -m "Update graphs and timestamp."
git push
