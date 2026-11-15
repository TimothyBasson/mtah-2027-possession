R="uv run --frozen --with scikit-learn==1.6.1 python logs_tune/train.py --sets skillcorner"
$R --tag sk_norm --norm speed --iters 100,300
$R --tag sk_in --indomain --iters 100,300
$R --tag sk_f3 --fold-seconds 3 --iters 100,300
$R --tag sk_raw --raw-target --iters 100,300
$R --tag sk_dropv --drop rel_v,relv,p_speed --iters 100,300
