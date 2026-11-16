#!/bin/sh
# The runs behind the BAR-ATH files, resumable (finished runs are skipped). About 2.5 h on 4 CPUs.
# c*: independent of Tim's model. t*: his HMM's labels and control posteriors as extra inputs.
# p / tp: adapted to the unseen provider (held-out folds) or to BAR-ATH (fold none) without labels.
J="sh alldata/learned/job.sh"
(sh alldata/learned/folds.sh tp1 --pseudo t1e) & $J cN none --drop-cols tim & wait
(sh alldata/learned/folds.sh tp1s1 --pseudo t1e --seed 1) & $J tN none & wait
$J cNs1 none --drop-cols tim --seed 1 & $J tNs1 none --seed 1 & wait
python3 alldata/learned/ens.py cNe cN cNs1 && python3 alldata/learned/ens.py tNe tN tNs1
$J pN none --drop-cols tim --pseudo cNe & $J pNs1 none --drop-cols tim --pseudo cNe --seed 1 & $J tpN none --pseudo tNe & $J tpNs1 none --pseudo tNe --seed 1 & wait
echo final queue done
