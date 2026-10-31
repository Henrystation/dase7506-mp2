"""Independent NumPy-only nonlinear V2 stack, for supplied input banks."""
import json
from pathlib import Path
import numpy as np
from equation_features import prepare_equations
from student_model import score_candidates
from neural_model import candidate_scores as neural_scores
from tree_model import tree_scores
from ensemble_model import pool_scores
from student_features import choose_from_scores
from v8_stack_features import prepare,features
from v8_cat_numeric import score_features

def center_valid(scores,groups):
    result=np.zeros_like(scores,dtype=np.float32)
    for i,gs in enumerate(groups):
        ids=[j for members in gs.values() for j in members]
        if ids:result[i,ids]=scores[i,ids]-scores[i,ids].mean()
    return result

def candidate_scores(bank,arrays,checkpoint):
    checkpoint=Path(checkpoint);config=json.loads((checkpoint/'stack_config.json').read_text())
    if config['version']!='v8_nested_stack_numpy':raise ValueError('Unexpected stack checkpoint')
    base=checkpoint/'v2';spec=json.loads((base/'ensemble.json').read_text())['members']
    if spec!=config['base_specification']:raise ValueError('Base configuration mismatch')
    _,groups,e,t=prepare_equations(bank,arrays)
    with np.load(base/'equations/weights.npz',allow_pickle=False) as z:ridge=score_candidates(np.concatenate([e,t],-1),z['weights'],z['scale'])
    neural,_,_=neural_scores(bank,arrays,base/'neural');tree=tree_scores(bank,arrays,base/'tree');raw=dict(equations=ridge,neural=neural,tree=tree)
    reference=np.zeros((len(bank),4),np.float32)
    for name,entry in spec.items():
        score=pool_scores(raw[name],groups,entry['pooling']);score-=score.mean(1,keepdims=True);reference+=entry['weight']*score/entry['oof_scale']
    data=prepare(bank,arrays);heads=[]
    for name in config['heads']:
        path=checkpoint/name
        with np.load(path/'projection.npz',allow_pickle=False) as z:projection={k:z[k] for k in z.files}
        x=features(data,raw,projection,spec);logits=score_features(x,path);heads.append(1/(1+np.exp(-np.clip(logits,-30,30))))
    probability=np.mean(heads,axis=0).astype(np.float32);pooled=pool_scores(probability,groups,'mean');weight=config['weight']
    scores=(1-weight)*center_valid(reference,groups)/config['reference_scale']+weight*center_valid(pooled,groups)/config['candidate_scale']
    return scores,groups

def predict(bank,arrays,checkpoint):
    scores,groups=candidate_scores(bank,arrays,checkpoint)
    return choose_from_scores(groups,scores,0.)
