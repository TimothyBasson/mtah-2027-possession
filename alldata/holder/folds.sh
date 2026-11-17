#!/bin/sh
# sh alldata/holder/folds.sh <fit.py|hfit.py> <tag> [options]: the three held-out folds side by side; resumable.
case $1 in hfit.py) py=alldata/holder/hfit.py;; *) py=alldata/learned/fit.py;; esac
tag=$2; shift 2
mkdir -p logs
for f in idsse metrica skillcorner; do
  [ -f data/cache/alldata/runs/$tag/$f/done ] && continue
  (python3 $py --fold $f --tag $tag "$@" > logs/alldata_${tag}_$f.log 2>&1 && touch data/cache/alldata/runs/$tag/$f/done) &
done
wait
