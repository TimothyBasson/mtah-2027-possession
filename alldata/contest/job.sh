#!/bin/sh
# sh alldata/contest/job.sh <tag> <fold> [cfit.py options]: one run, skipped if it already finished.
tag=$1; fold=$2; shift 2
mkdir -p logs
[ -f data/cache/alldata/runs/$tag/$fold/done ] && exit 0
python3 alldata/contest/cfit.py --fold $fold --tag $tag --drop-cols tim --share skillcorner=0.25 "$@" > logs/contest_${tag}_$fold.log 2>&1 && touch data/cache/alldata/runs/$tag/$fold/done
