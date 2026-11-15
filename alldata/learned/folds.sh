#!/bin/sh
# sh alldata/learned/folds.sh <tag> [train.py options]: the three held-out folds side by side.
tag=$1; shift
mkdir -p logs
for f in idsse metrica skillcorner; do
  [ -f data/cache/alldata/runs/$tag/$f/done ] && continue
  (python3 alldata/learned/fit.py --fold $f --tag $tag "$@" > logs/alldata_${tag}_$f.log 2>&1 && touch data/cache/alldata/runs/$tag/$f/done) &
done
wait
