#!/bin/bash

DOCKER_IMAGE=ghcr.io/rutherford-nj/data-committee.covid-data:latest

# Run scraper.
echo "::group::Fetch Data"
docker run -v `pwd`:/work --entrypoint="/work/data/fetch.sh" $DOCKER_IMAGE
echo "::endgroup::"

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

docker run -v `pwd`:/work --entrypoint="/work/run.sh" --env COVID_SMA_WIN=7 $DOCKER_IMAGE
docker run -v `pwd`:/work --entrypoint="/work/run.sh" $DOCKER_IMAGE

# Run svgo to optimize SVG images.
echo "::group::Optimize SVGs"
docker run -v `pwd`:/work --entrypoint="/work/data/svgo.sh" $DOCKER_IMAGE
echo "::endgroup::"

# Ensure the files needed for rutherfordboronj.com are available, or exit.
[ -f docs/total_cases_per_100K_14d_SMA.svg ] && [ -f docs/new_cases_per_100K_14d_SMA.svg ] || exit 1

# Create timestamp file.
TZ=America/New_York date +"%A %B %d at %l:%M%P" | tee docs/last_updated

# Commit web-ready files and push.
git add docs
git commit -m "Update graphs and timestamp."
git push
