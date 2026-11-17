#!/bin/sh
# sh alldata/selftrain/job.sh <tag> <fold> [selftrain.py options]: one run, skipped if it already finished.
tag=$1; fold=$2; shift 2
mkdir -p logs
[ -f data/cache/alldata/runs/$tag/$fold/done ] && exit 0
python3 alldata/selftrain/selftrain.py --fold $fold --tag $tag "$@" > logs/st_${tag}_$fold.log 2>&1 && touch data/cache/alldata/runs/$tag/$fold/done
