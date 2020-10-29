#!/bin/sh

cd /work/docs
npm install -g svgo
svgo *svg
