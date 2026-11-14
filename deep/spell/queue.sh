#!/bin/sh
# Run head.py variants over the three folds, four at a time, one thread each; results in logs/heads.log
#   sh deep/spell/queue.sh "NAME|options" "NAME2|options" ...   (SEEDS="0 1" for more seeds)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
for spec in "$@"; do
  name=${spec%%|*}; opts=${spec#*|}
  for seed in ${SEEDS:-0}; do
    for f in idsse metrica skillcorner; do
      grep -q "^RESULT $name seed $seed $f " logs/heads.log 2>/dev/null || echo "--fold $f --name $name --seed $seed $opts"
    done
  done
done | SCRIPT=${SCRIPT:-head} xargs -P 4 -L 1 sh -c '.venv/bin/python deep/spell/${SCRIPT:-head}.py "$@" 2>>logs/heads.err | grep RESULT >> logs/heads.log' sh
