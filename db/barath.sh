set -e
R="uv run --frozen --with torch==2.15.0 python"
for a in 0 1 2 4; do
  $R -m hmm.decode --ahead $a --out evaluation/results/db_ahead$a.csv > logs_db/barath_ahead$a.log 2>&1
done
$R evaluation/run_evaluation.py > logs_db/barath_eval.log 2>&1
