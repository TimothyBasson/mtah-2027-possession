#!/bin/sh
# The runs behind the tables and the BAR-ATH file, resumable (finished runs are skipped); about
# 1.5 h on 4 CPUs. s*: trained on outside balls made to look like BAR-ATH's, independent of Tim's
# model. ps*: adapted without labels to the unseen provider (held-out folds) or to BAR-ATH (N).
J="sh alldata/learned/job.sh"
D="--drop-cols tim"
(sh alldata/learned/folds.sh s1 $D) & $J sN none $D & wait
(sh alldata/learned/folds.sh s1s1 $D --seed 1) & $J sNs1 none $D --seed 1 & wait
python3 alldata/learned/ens.py s1e s1 s1s1 && python3 alldata/learned/ens.py sNe sN sNs1
(sh alldata/learned/folds.sh ps1 $D --pseudo s1e) & $J psN none $D --pseudo sNe & wait
(sh alldata/learned/folds.sh ps1s1 $D --pseudo s1e --seed 1) & $J psNs1 none $D --pseudo sNe --seed 1 & wait
echo final queue done
