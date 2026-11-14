"""deep/'s held-out matches scored against the outside labels as they are and folded towards the
challenge's controlled-possession rule (external/control.py), with the shipped decode in every
fold, for comparison with external/score_control.py's rows for hmm.

    python deep/folded.py [--tag v2]
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else 'v2'
os.chdir(ROOT); sys.path[:0]=[ROOT,os.path.join(ROOT,'deep'),os.path.join(ROOT,'external'),os.path.join(ROOT,'evaluation')]
import possession_metrics as pm, decode
import score_hmm as sh
from control import CONTROL_SECONDS, control_labels
from concurrent.futures import ProcessPoolExecutor
def one(f):
    name=os.path.basename(f)[:-4]; p=np.load(f); d=sh.load(os.path.join(ROOT,'data','external',name+'.npz'))
    spans=[tuple(h) for h in d['halves']]
    pred=pm.label_codes(decode.labels_from([p[f'p{i}'] for i in range(len(spans))],spans,len(d['labels']),cost=4,min_team=1,min_dead=1))
    return name,[pm.evaluate(pm.label_codes(v),pred,bootstrap_n=0)['spell_f1_mean'] for v in (d['labels'],control_labels(d,CONTROL_SECONDS))]
if __name__=='__main__':
    res={}
    with ProcessPoolExecutor(4) as ex:
        for fold in ('idsse','metrica','skillcorner'):
            r=list(ex.map(one,sorted(glob.glob(f'data/cache/deep/{TAG}/{fold}/{fold}_*.npz'))))
            res[fold]=np.mean([x[1] for x in r],0); print(fold,'raw %.3f folded %.3f'%tuple(res[fold]),flush=True)
    print('mean raw %.3f folded %.3f'%tuple(np.mean(list(res.values()),0)))
