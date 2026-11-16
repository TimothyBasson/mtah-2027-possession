#!/bin/sh
# The comparison: fit.py (w*) against hfit.py (k*), SkillCorner at a quarter of the training
# windows, two seeds, then both adapted to the unseen provider without labels (pw*, pk*).
F="sh alldata/holder/folds.sh"
S="--share skillcorner=0.25"
$F fit.py w1 --drop-cols tim $S
$F hfit.py k1 $S
$F fit.py w1s1 --drop-cols tim $S --seed 1
$F hfit.py k1s1 $S --seed 1
python3 alldata/learned/ens.py w1e w1 w1s1 && python3 alldata/learned/ens.py k1e k1 k1s1
$F fit.py pw1 --drop-cols tim $S --pseudo w1e
$F hfit.py pk1 $S --pseudo k1e
$F fit.py pw1s1 --drop-cols tim $S --pseudo w1e --seed 1
$F hfit.py pk1s1 $S --pseudo k1e --seed 1
python3 alldata/learned/ens.py pw1e pw1 pw1s1 && python3 alldata/learned/ens.py pk1e pk1 pk1s1
echo holder queue done
