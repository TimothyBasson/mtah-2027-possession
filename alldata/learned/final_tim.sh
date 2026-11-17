#!/bin/sh
# The with-Tim runs on BAR-ATH-like balls (after tim_barlike.py and build.py), resumable.
# ts*: his HMM's labels and control posteriors as extra inputs. tps*: adapted without labels.
J="sh alldata/learned/job.sh"
(sh alldata/learned/folds.sh ts1) & $J tsN none & wait
(sh alldata/learned/folds.sh ts1s1 --seed 1) & $J tsNs1 none --seed 1 & wait
python3 alldata/learned/ens.py ts1e ts1 ts1s1 && python3 alldata/learned/ens.py tsNe tsN tsNs1
(sh alldata/learned/folds.sh tps1 --pseudo ts1e) & $J tpsN none --pseudo tsNe & wait
(sh alldata/learned/folds.sh tps1s1 --pseudo ts1e --seed 1) & $J tpsNs1 none --pseudo tsNe --seed 1 & wait
echo final queue done
