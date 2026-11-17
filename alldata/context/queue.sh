#!/bin/bash
# Context-pass fits four at a time from a job list (lines: tag fold [args]); skips finished ones.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
mkdir -p logs
job() { tag=$1; fold=$2; shift 2
  d=data/cache/alldata/runs/$tag/$fold
  [ -f $d/done ] && exit 0
  OMP_NUM_THREADS=1 python alldata/context/ctx.py --tag $tag --fold $fold "$@" > logs/ctx_${tag}_$fold.log 2>&1 && touch $d/done; }
export -f job
xargs -P 4 -L 1 bash -c 'job "$@"' _ < "$1"
