#!/bin/sh
# Both together: SkillCorner drawn for a quarter of the training windows and Tim v8 as inputs; resumable.
# tw*: base, tpw*: adapted without labels.
J="sh alldata/learned/job.sh"
D="--share skillcorner=0.25"
(sh alldata/learned/folds.sh tw1 $D) & $J twN none $D & wait
(sh alldata/learned/folds.sh tw1s1 $D --seed 1) & $J twNs1 none $D --seed 1 & wait
python3 alldata/learned/ens.py tw1e tw1 tw1s1 && python3 alldata/learned/ens.py twNe twN twNs1
(sh alldata/learned/folds.sh tpw1 $D --pseudo tw1e) & $J tpwN none $D --pseudo twNe & wait
(sh alldata/learned/folds.sh tpw1s1 $D --pseudo tw1e --seed 1) & $J tpwNs1 none $D --pseudo twNe --seed 1 & wait
echo both queue done
