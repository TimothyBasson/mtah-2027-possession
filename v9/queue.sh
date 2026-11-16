#!/bin/sh
# Run v9/train.py jobs three at a time, one thread each; finished folds are skipped, unfinished resume.
#   sh v9/queue.sh v9/jobs.txt      # lines of: TAG FOLD [train.py options]
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
PY=${PY:-python}
one() {
  tag=$1; fold=$2; shift 2
  [ -f "data/cache/v9/$tag/$fold/net.pt" ] || $PY v9/train.py --fold "$fold" --tag "$tag" "$@" >> "logs/v9_${tag}_$fold.log" 2>&1
}
case "$1" in
  --one) shift; one "$@" ;;
  *) grep -v '^#' "$1" | grep . | xargs -P "${JOBS:-3}" -L 1 sh "$0" --one ;;
esac
