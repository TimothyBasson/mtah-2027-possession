R="uv run --frozen --with scikit-learn==1.6.1 python logs_tune/train.py --sets skillcorner"
$R --tag sk_base --iters 25,50,100,200,300
$R --tag sk_l15 --leaves 15 --min-leaf 1000 --iters 50,100,200,300
$R --tag sk_mf3 --max-features 0.3 --iters 100,200,300
$R --tag sk_lr03 --lr 0.03 --iters 100,300,600
$R --tag sk_f3 --fold-seconds 3 --iters 100,300
$R --tag sk_raw --raw-target --iters 100,300
$R --tag sk_dropv --drop rel_v,relv,p_speed --iters 100,300
