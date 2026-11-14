#!/bin/sh
# Run a list of training jobs four at a time, one thread each. Safe to rerun: finished jobs are
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
grep -v '^#' "$1" | grep . | while read -r line; do
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 20; done
  # shellcheck disable=SC2086
  one $line &
done
wait
