#!/bin/sh
# Train the three held-out folds side by side, one thread each. Safe to rerun: each fold resumes.
#   sh deep/run_folds.sh TAG [train.py options]
cd "$(dirname "$0")/.." || exit 1
tag=$1; shift
mkdir -p logs
for f in idsse metrica skillcorner; do
  [ -f "data/cache/deep/$tag/$f/net.pt" ] && continue
  pgrep -f "train.py --fold $f .*--tag $tag" >/dev/null && continue
  nohup .venv/bin/python deep/train.py --fold $f --threads 1 --tag "$tag" "$@" >> "logs/${tag}_$f.log" 2>&1 &
done
