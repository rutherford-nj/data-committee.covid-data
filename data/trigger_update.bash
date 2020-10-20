#!/bin/bash

if [ -z ${GITHUB_PAT+x} ]; then echo "Set GITHUB_PAT before running."; exit 1; fi

curl -H "Accept: application/vnd.github.v3+json" \
    -H "Authorization: token $GITHUB_PAT" \
    --request POST \
    --data '{"event_type": "update-data"}' \
    https://api.github.com/repos/grecine/covid_data/dispatches
