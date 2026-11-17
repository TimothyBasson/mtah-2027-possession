set -e
R="uv run --frozen --with torch==2.15.0 python"
$R -m hmm.decode --out evaluation/results/kp_base.csv > logs_kp/barath_base.log 2>&1
sha256sum evaluation/results/kp_base.csv >> logs_kp/barath_base.log
for spec in "box 4" "box 8" "box 16" "next 4" "next 8"; do
  set -- $spec
  $R -m hmm.decode --keeper $2 --keeper-ends $1 --out evaluation/results/kp_$1$2.csv > logs_kp/barath_$1$2.log 2>&1
done
$R evaluation/run_evaluation.py > logs_kp/barath_eval.log 2>&1
