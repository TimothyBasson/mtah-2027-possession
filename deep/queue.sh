#!/bin/sh
# Run a list of training jobs three at a time (JOBS=n to change; each holds the 3 GB tensor cache), one thread each. Safe to rerun: finished jobs are
# skipped and unfinished ones resume from their checkpoint. Each finished fold is copied to
# /mnt/project-files/deep/ when that folder exists, because the cloud container does not last.
#   sh deep/queue.sh deep/jobs.txt      # lines of: TAG FOLD [train.py options]
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
one() {
  tag=$1; fold=$2; shift 2
  out="data/cache/deep/$tag/$fold"
  if [ ! -f "$out/net.pt" ]; then
    python deep/train.py --fold "$fold" --threads 1 --tag "$tag" "$@" >> "logs/${tag}_$fold.log" 2>&1 || exit 1
  fi
  if [ -d /mnt/project-files/deep/ ]; then
    mkdir -p "/mnt/project-files/deep/$tag/$fold" && cp "$out"/*.npz "$out/net.pt" "/mnt/project-files/deep/$tag/$fold/"
  fi
}
case "$1" in
  --one) shift; one "$@" ;;
  *) grep -v '^#' "$1" | grep . | xargs -P "${JOBS:-3}" -L 1 sh "$0" --one ;;
esac
