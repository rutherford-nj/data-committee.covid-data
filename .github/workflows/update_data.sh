#!/bin/sh

set -x

cd data && go run *.go && ../
git config user.name github-actions
git config user.email github-actions@github.com
timestamp=$(TZ=America/New_York date)
git commit -am "Latest data: ${timestamp}." || exit 0
docker build -t rutherford_covid_image .
docker run -v `pwd`:/work --entrypoint="/work/run.sh" rutherford_covid_image
docker run -v `pwd`:/work --entrypoint="/work/data/svgo.sh" node:15.0.1-alpine3.12
TZ=America/New_York date +"%A %B %d at %l:%M%P" | tee docs/last_updated
git commit -am "Update graphs and timestamp."
git push
