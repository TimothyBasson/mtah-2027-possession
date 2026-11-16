#!/bin/sh
# SkillCorner drawn for a quarter of the training windows (its labels are the noisy ones, PR 42),
# independent of Tim's model; resumable. w*: base, pw*: adapted without labels.
J="sh alldata/learned/job.sh"
D="--drop-cols tim --share skillcorner=0.25"
(sh alldata/learned/folds.sh w1 $D) & $J wN none $D & wait
(sh alldata/learned/folds.sh w1s1 $D --seed 1) & $J wNs1 none $D --seed 1 & wait
python3 alldata/learned/ens.py w1e w1 w1s1 && python3 alldata/learned/ens.py wNe wN wNs1
(sh alldata/learned/folds.sh pw1 $D --pseudo w1e) & $J pwN none $D --pseudo wNe & wait
(sh alldata/learned/folds.sh pw1s1 $D --pseudo w1e --seed 1) & $J pwNs1 none $D --pseudo wNe --seed 1 & wait
echo share queue done
