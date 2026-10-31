"""Rebuild the fixed final V8 model using official training data only.

From code/:
    python train_v8.py --checkpoint ../reproduction/model_bundle \
        --record ../reproduction/training_record.json --validate

No saved predictions, original workspace caches, or pretrained model files are
read. All three base models, five-fold base OOF, PCA, and the PairLogit head are
fitted afresh. --validate evaluates only after every fitted asset is finalized;
validation labels cannot affect training or change the fixed configuration.

The historical V2 OOF normalizers and final mixture scales are fixed recipe
constants. They are preserved rather than reestimated from the new five-fold
OOF, because changing them would define a different V8 feature representation.
The final user choice of V8 followed an authorized retrospective test review;
this reproduction does not claim a new pre-test selection or use test labels.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from catboost import CatBoostRanker, Pool

from common import DATA, check_data, load_inputs, read_jsonl
from student_features import prepare_inputs, group_correctness, choose_from_scores
from student_model import fit_ridge, score_candidates
from equation_features import prepare_equations
from tree_features import prepare_extra, feature_array
from train_tree_v2 import fit as fit_tree, export_model as export_tree
from experiments.v2_neural import train as fit_neural, export as export_neural, preprocess
from neural_model import _gelu
from v8_stack_features import prepare, subset, fit_projection, features
from v8_cat_numeric import export_numeric, score_features

CODE = Path(__file__).resolve().parent
NAMES = ('equations','neural','tree')
BASE_SPEC = {
    'equations':dict(weight=.5,pooling='max',oof_scale=.24336877465248108),
    'neural':dict(weight=.25,pooling='max',oof_scale=.9408119916915894),
    'tree':dict(weight=.25,pooling='mean',oof_scale=1.1752545833587646),
}
NEURAL = dict(hidden=128,dropout=.35,lr=.0005,wd=.15,batch_size=128,
              epochs=60,seed=20261030,base_scale=4.,penalty=.02,bce=.2,
              mode='both',loss='soft',device='cpu')
NEURAL_FIXED_EPOCHS = 5
OOF_SEED = 20261031
HEAD = dict(loss_function='PairLogit',depth=4,iterations=600,learning_rate=.03,
            l2_leaf_reg=20,random_seed=20261104,thread_count=4,task_type='CPU',
            bootstrap_type='Bernoulli',subsample=.8,random_strength=1,
            border_count=64,grow_policy='SymmetricTree',allow_writing_files=False,
            verbose=False)
SOURCE_FILES = (
    'train_v8.py','common.py','student_features.py','student_model.py',
    'equation_features.py','tree_features.py','train_tree_v2.py',
    'experiments/v2_neural.py','neural_model.py','tree_model.py',
    'ensemble_model.py','trajectory_features_v6.py','v8_stack_features.py',
    'v8_cat_numeric.py','v8_stack_model.py',
)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def fit_base_models(data, indices, query=None, output=None):
    """Fit one base triplet with fixed epochs and no held-side labels."""
    groups=[data['groups'][int(i)] for i in indices]
    truth=data['truth'][indices]
    rw,rs,_=fit_ridge(data['ridge'][indices],truth,groups,1.)
    # The legacy helper requires a scoring argument even with fixed epochs.
    # Supply a fit-side row only; its diagnostic vote never affects parameters.
    dummy=indices[:1]
    neural=fit_neural(NEURAL,data['neural'][indices],truth,groups,
        data['neural'][dummy],data['truth'][dummy],[data['groups'][int(dummy[0])]],
        epochs=NEURAL_FIXED_EPOCHS)
    hard_indices=indices[np.asarray([len(group)>=2 for group in groups])]
    tree=fit_tree(data['tree'][hard_indices],data['truth'][hard_indices])
    if output is not None:
        output=Path(output);(output/'equations').mkdir(parents=True,exist_ok=True)
        np.savez(output/'equations/weights.npz',weights=rw,scale=rs)
        export_neural(neural,output/'neural');export_tree(tree,output/'tree')
        config=dict(version='mp2_answer_group_ensemble_v2',members=BASE_SPEC,
                    vote_strength=0.,cv_seed=20261030,threads=4)
        (output/'ensemble.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf8')
    if query is None:
        return None
    x,_=preprocess(data['neural'][indices],data['neural'][query],NEURAL,neural['stats'])
    p=neural['params']
    hidden=_gelu(x@p['fc1.weight'].T+p['fc1.bias'])
    hidden=_gelu(hidden@p['fc2.weight'].T+p['fc2.bias'])
    nscore=NEURAL['base_scale']*score_candidates(data['neural'][query],neural['ridge_weights'],neural['ridge_scale'])+(hidden@p['out.weight'].T+p['out.bias']).squeeze(-1)
    result=dict(equations=score_candidates(data['ridge'][query],rw,rs),neural=nscore,
        tree=tree.predict(data['tree'][query].reshape(-1,data['tree'].shape[-1]),raw_score=True).reshape(-1,4))
    if any(score.shape!=(len(query),4) or not np.isfinite(score).all() for score in result.values()):
        raise AssertionError('Base OOF prediction shape/finiteness failed')
    return result


def validation_only(checkpoint,data_dir):
    """Called after training/export only; no parameter or model selection."""
    from v8_stack_model import predict
    tick=time.perf_counter()
    check_data(data_dir,splits=['validation'])
    bank,arrays=load_inputs('validation',data_dir)
    references=read_jsonl(data_dir/'gsm8k_validation.jsonl')
    answers,groups,_,_=prepare_inputs(bank,arrays)
    if [(r['id'],r['question']) for r in bank]!=[(r['id'],r['question']) for r in references]:
        raise AssertionError('Validation alignment differs')
    truth=group_correctness(answers,references)
    choices=np.asarray(predict(bank,arrays,checkpoint))
    hard=np.flatnonzero([len(group)>=2 for group in groups])
    hard_correct=int(truth[hard,choices[hard]].sum())
    all_correct=int(truth[np.arange(len(bank)),choices].sum())
    return dict(split='validation',hard_correct=hard_correct,hard_n=len(hard),
                all_correct=all_correct,all_n=len(bank),seconds=time.perf_counter()-tick,
                historical_hard_correct=313,historical_all_correct=756,
                matches_historical=(hard_correct==313 and all_correct==756),
                used_for_training_or_selection=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--validate',action='store_true',help='Evaluate validation only after training; no fitting or selection')
    args=parser.parse_args()
    output=args.checkpoint.resolve();record_path=args.record.resolve();data_dir=args.data.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite a nonempty checkpoint; choose a fresh directory')
    if record_path.exists():
        raise ValueError('Refusing to overwrite an existing training record')
    if record_path.is_relative_to(output):
        raise ValueError('Keep development records outside the inference checkpoint')
    output.mkdir(parents=True,exist_ok=True);record_path.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    started=time.perf_counter();check_data(data_dir,splits=['train'])
    bank,arrays=load_inputs('train',data_dir)
    references=read_jsonl(data_dir/'gsm8k_train.jsonl')
    if [(r['id'],r['question']) for r in bank]!=[(r['id'],r['question']) for r in references]:
        raise AssertionError('Training alignment differs')
    answers,groups,embedding,basic=prepare_inputs(bank,arrays)
    truth=group_correctness(answers,references)
    if not truth.any(axis=1).all():
        raise AssertionError('Training row has no correct candidate')
    _,_,_,equations=prepare_equations(bank,arrays)
    _,_,_,tree_text=prepare_extra(bank,arrays)
    data=dict(groups=groups,truth=truth,ridge=np.concatenate([embedding,equations],axis=-1),
              neural=np.concatenate([embedding,basic],axis=-1),
              tree=feature_array(embedding,tree_text,groups,'raw'))
    hard=np.flatnonzero([len(group)>=2 for group in groups]);all_indices=np.arange(len(bank))
    tick=time.perf_counter();fit_base_models(data,all_indices,output=output/'v2')
    full_base_seconds=time.perf_counter()-tick
    print(json.dumps(dict(stage='full_base_models',seconds=full_base_seconds)),flush=True)
    oof={name:np.full((len(hard),4),np.nan,np.float32) for name in NAMES}
    fold_records=[];coverage=[]
    for fold,held in enumerate(np.array_split(np.random.default_rng(OOF_SEED).permutation(hard),5)):
        tick=time.perf_counter();fitted=np.setdiff1d(all_indices,held)
        if np.intersect1d(fitted,held).size:
            raise AssertionError('OOF training/held overlap')
        prediction=fit_base_models(data,fitted,query=held)
        positions=np.searchsorted(hard,held)
        if not np.array_equal(hard[positions],held):
            raise AssertionError('OOF row alignment differs')
        for name in NAMES:oof[name][positions]=prediction[name]
        coverage.extend(held.tolist())
        record=dict(fold=fold,fit_indices=fitted.tolist(),held_indices=held.tolist(),seconds=time.perf_counter()-tick)
        fold_records.append(record)
        print(json.dumps(dict(stage='base_oof',fold=fold+1,folds=5,seconds=record['seconds'])),flush=True)
    if sorted(coverage)!=hard.tolist() or len(set(coverage))!=len(hard) or not all(np.isfinite(s).all() for s in oof.values()):
        raise AssertionError('OOF coverage/finiteness failed')
    # Evidence is saved outside the checkpoint and is never loaded to fit this run.
    evidence_file=record_path.parent/(record_path.stem+'_base_oof.npz')
    if evidence_file.exists():raise ValueError('Refusing to overwrite existing OOF evidence')
    np.savez(evidence_file,hard_indices=hard,**oof)
    tick=time.perf_counter();stack_data=prepare(bank,arrays)
    projection=fit_projection(stack_data,hard)
    x=features(subset(stack_data,hard),oof,projection,BASE_SPEC)
    head_groups=[groups[int(i)] for i in hard];labels=truth[hard]
    pairs=[];pair_weights=[]
    for row,answer_groups in enumerate(head_groups):
        valid=[index for members in answer_groups.values() for index in members]
        good=[index for index in valid if labels[row,index]]
        bad=[index for index in valid if not labels[row,index]]
        if not good or not bad:raise AssertionError('PairLogit requires both classes per hard row')
        for positive in good:
            for negative in bad:
                pairs.append([4*row+positive,4*row+negative]);pair_weights.append(1/(len(good)*len(bad)))
    pool=Pool(x.reshape(-1,x.shape[-1]),labels.ravel().astype(int),
              group_id=np.repeat(np.arange(len(hard)),4),pairs=pairs,pairs_weight=pair_weights)
    head=CatBoostRanker(**HEAD).fit(pool)
    head_output=output/'pair4';export_numeric(head,head_output)
    np.savez(head_output/'projection.npz',**projection)
    expected=head.predict(x[:64].reshape(-1,x.shape[-1]),thread_count=4).reshape(64,4)
    replay=score_features(x[:64],head_output)
    replay_error=float(np.max(np.abs(expected-replay)))
    np.testing.assert_allclose(replay,expected,atol=2e-6,rtol=2e-6)
    head_seconds=time.perf_counter()-tick
    config=dict(version='v8_nested_stack_numpy',heads=['pair4'],weight=1,
                candidate_scale=.2744070887565613,reference_scale=.8949364423751831,
                base_specification=BASE_SPEC,status='Reproduced fixed final V8 recipe from official train only',
                selection='Final V8 choice follows authorized retrospective comparison; no new selection in this reproduction')
    (output/'stack_config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf8')
    training_seconds=time.perf_counter()-started
    record=dict(recipe='fixed_v8_pair4',data_used_for_fit='official train only',
        training_questions=len(bank),training_hard_questions=len(hard),
        full_base_seconds=full_base_seconds,head_and_projection_seconds=head_seconds,
        training_seconds=training_seconds,neural_config=NEURAL,neural_fixed_epochs=NEURAL_FIXED_EPOCHS,
        head_config=HEAD,base_specification=BASE_SPEC,base_oof_seed=OOF_SEED,folds=fold_records,
        scalar_provenance='Historical train-derived V2 OOF normalizers and first-stage V8 mixture scales are frozen recipe constants; no new validation calibration.',
        nested_outer_evaluation_repeated=False,head_input='Fresh five-fold base OOF, never in-sample base scores',
        fit_query_labels_used=False,test_accessed=False,audit_accessed=False,
        validation_used_for_training_or_selection=False,hyperparameter_search=False,
        numpy_head_replay_max_error=replay_error,threads=4,device='cpu',
        versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','torch','lightgbm','catboost')},
        input_sha256={name:sha(data_dir/name) for name in ('gsm8k_train.jsonl','candidates/gsm8k_train.jsonl','candidates/gsm8k_train.npz')},
        source_sha256={name:sha(CODE/name) for name in SOURCE_FILES},
        checkpoint_sha256={str(path.relative_to(output)):sha(path) for path in sorted(output.rglob('*')) if path.is_file()},
        base_oof_file=evidence_file.name,base_oof_sha256=sha(evidence_file))
    # No fitted asset is changed after this point, including in validation mode.
    if args.validate:record['validation']=validation_only(output,data_dir)
    record_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:record[key] for key in ('recipe','training_seconds','numpy_head_replay_max_error')}
                     | {'validation':record.get('validation'),'checkpoint':str(output),'record':str(record_path)},indent=2),flush=True)


if __name__=='__main__':main()
