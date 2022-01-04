#!/bin/bash

cd data
python3 update_rutherford.py $1
cd ..

# Setup local git config.
git config --global user.name github-actions
git config --global user.email github-actions@github.com

# Commit data changes. For scheduled runs, bail out of the workflow if there hasn't been a change in the data.
timestamp=$(TZ=America/New_York date)
git commit -am "Latest data: ${timestamp}."
git push
