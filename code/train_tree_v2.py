"""Reproduce the fixed raw-feature tree component using official training only.

LightGBM is a training dependency. Exported inference uses NumPy only.
"""
import argparse
import json
import os
from pathlib import Path
import time
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="4"
import numpy as np
import lightgbm as lgb
from common import DATA,load_inputs,read_jsonl
from student_features import group_correctness
from tree_features import prepare_extra,feature_array

PARAMETERS=dict(objective="lambdarank",metric="None",num_leaves=15,learning_rate=.04,
                min_data_in_leaf=35,lambda_l2=10.,feature_fraction=.8,bagging_fraction=.9,
                bagging_freq=1,verbosity=-1,num_threads=4,seed=7506,deterministic=True,force_col_wise=True)
ROUNDS=600

def fit(features,labels):
    """Fit the frozen tree recipe on caller-supplied training groups only."""
    features=np.asarray(features,dtype=np.float32)
    labels=np.asarray(labels,dtype=bool)
    if features.ndim!=3 or features.shape[1]!=4 or labels.shape!=features.shape[:2]:
        raise ValueError("Expected [N,4,F] features and [N,4] correctness")
    ds=lgb.Dataset(features.reshape(-1,features.shape[-1]),label=labels.ravel().astype(int),group=[4]*len(features),free_raw_data=True)
    return lgb.train(PARAMETERS,ds,num_boost_round=ROUNDS)

def export_model(model,output_dir):
    """Export every learned numeric tree to the NumPy inference format."""
    export_numeric_trees(model.dump_model(),output_dir)

def export_numeric_trees(dump,output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    fs=[];ls=[];rs=[];ts=[];vs=[];ds=[];roots=[]
    def add(node):
        index=len(fs)
        fs.append(-1);ls.append(-1);rs.append(-1);ts.append(0.);vs.append(0.);ds.append(False)
        if "leaf_value" in node:
            vs[index]=node["leaf_value"]
        else:
            if node["decision_type"]!="<=":raise ValueError("Only numeric trees are supported")
            fs[index]=node["split_feature"];ts[index]=float(node["threshold"]);ds[index]=node["default_left"]
            ls[index]=add(node["left_child"]);rs[index]=add(node["right_child"])
        return index
    for tree in dump["tree_info"]:roots.append(add(tree["tree_structure"]))
    np.savez(output_dir/"tree_weights.npz",feature=np.array(fs,np.int32),left=np.array(ls,np.int32),right=np.array(rs,np.int32),
             threshold=np.array(ts,np.float64),value=np.array(vs,np.float64),default_left=np.array(ds,bool),roots=np.array(roots,np.int32))
    cfg=dict(version="mp2_numeric_gbdt_v1",mode="raw",n_features=dump["max_feature_idx"]+1,num_trees=len(roots))
    (output_dir/"tree_config.json").write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--record",type=Path,required=True);a=p.parse_args()
    start=time.perf_counter()
    bank,arrays=load_inputs("train")
    answers,groups,emb,text=prepare_extra(bank,arrays)
    refs=read_jsonl(DATA/"gsm8k_train.jsonl")
    if [r["id"] for r in refs]!=[r["id"] for r in bank]:raise ValueError("Training reference alignment failed")
    labels=group_correctness(answers,refs)
    hard=np.flatnonzero([len(g)>=2 for g in groups])
    features=feature_array(emb,text,groups,"raw")[hard]
    model=fit(features,labels[hard])
    export_model(model,a.checkpoint)
    record=dict(method="raw_feature_lambdarank_gbdt_v2",parameters=PARAMETERS,rounds=ROUNDS,
                training_groups=len(bank),training_hard_groups=len(hard),n_features=features.shape[-1],
                seed=7506,numpy_version=np.__version__,lightgbm_version=lgb.__version__,
                training_seconds=time.perf_counter()-start,
                supervision="Only official train inputs and references; no validation or test references loaded")
    a.record.parent.mkdir(parents=True,exist_ok=True);a.record.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(record,indent=2))

if __name__=="__main__":main()
