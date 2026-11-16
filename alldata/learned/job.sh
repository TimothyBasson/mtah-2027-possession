#!/bin/sh
# sh alldata/learned/job.sh <tag> <fold> [fit.py options]: one run, skipped if it already finished.
tag=$1; fold=$2; shift 2
mkdir -p logs
[ -f data/cache/alldata/runs/$tag/$fold/done ] && exit 0
python3 alldata/learned/fit.py --fold $fold --tag $tag "$@" > logs/alldata_${tag}_$fold.log 2>&1 && touch data/cache/alldata/runs/$tag/$fold/done
