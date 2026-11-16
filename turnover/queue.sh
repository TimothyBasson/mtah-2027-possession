#!/bin/sh
# sh turnover/queue.sh TAG [ttrain.py options]: the three folds side by side, one thread each, resumable.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
tag=$1; shift
for fold in idsse metrica skillcorner; do
  [ -f "data/cache/turnover/$tag/$fold/done" ] || (python turnover/ttrain.py --fold $fold --tag "$tag" "$@" >> "logs/turn_${tag}_$fold.log" 2>&1 && touch "data/cache/turnover/$tag/$fold/done") &
done
wait
