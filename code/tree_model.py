"""NumPy-only numeric-tree evaluation; no training or reference access."""
import json
from pathlib import Path
import numpy as np
from tree_features import prepare_extra,feature_array

def score_features(features,checkpoint):
    checkpoint=Path(checkpoint)
    with np.load(checkpoint/"tree_weights.npz",allow_pickle=False) as pkg:
        f=pkg["feature"];l=pkg["left"];r=pkg["right"];t=pkg["threshold"]
        v=pkg["value"];d=pkg["default_left"];roots=pkg["roots"]
        x=features.reshape(-1,features.shape[-1]);total=np.zeros(len(x),np.float64)
        for root in roots:
            node=np.full(len(x),root,np.int32)
            active=f[node]>=0
            while active.any():
                ix=np.flatnonzero(active);current=node[ix];values=x[ix,f[current]]
                goleft=np.where(np.isnan(values),d[current],values<=t[current])
                node[ix]=np.where(goleft,l[current],r[current]);active=f[node]>=0
            total+=v[node]
    return total.reshape(features.shape[:2]).astype(np.float32)

def tree_scores(bank,arrays,checkpoint):
    cfg=json.loads((Path(checkpoint)/"tree_config.json").read_text())
    if cfg["version"]!="mp2_numeric_gbdt_v1":raise ValueError("Unsupported tree version")
    _,groups,emb,text=prepare_extra(bank,arrays)
    features=feature_array(emb,text,groups,cfg["mode"])
    if features.shape[-1]!=cfg["n_features"]:raise ValueError("Tree feature dimension mismatch")
    return score_features(features,checkpoint)
