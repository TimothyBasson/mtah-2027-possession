#!/bin/sh
# sh alldata/robust/queue.sh <jobs file> [parallel]: lines "tag fold fit.py-options", four at a
# time by default, finished runs skipped (alldata/learned/job.sh).
grep -v '^#' "$1" | grep . | xargs -P "${2:-4}" -L 1 sh alldata/learned/job.sh
