#!/bin/sh
# alldata/contest/job.sh, plus a lock so two queues never run the same job.
tag=$1; fold=$2
d=data/cache/alldata/runs/$tag/$fold
mkdir -p $d
[ -f $d/done ] && exit 0
mkdir $d/.lock 2>/dev/null || exit 0
sh alldata/contest/job.sh "$@"
rmdir $d/.lock
