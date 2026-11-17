# sh alldata/briefsafe/queue.sh <jobs.txt> [parallel]: each line "<tag> <fold> [cfit options]"; resumable.
P=${2:-4}
. .venv/bin/activate
grep -v '^#' "$1" | xargs -P "$P" -L 1 sh alldata/briefsafe/job.sh
