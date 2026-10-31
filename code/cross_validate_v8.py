"""Fresh train-only 5-outer x 3-inner evaluation of the fixed V8 recipe.

    python cross_validate_v8.py --seed 20261031 --output ../nested_cv_20261031
    python cross_validate_v8.py --seed 20261101 --output ../nested_cv_20261101

Every outer reference model, inner base model, PCA and PairLogit head is fitted
afresh. No prior checkpoint, OOF file or workspace cache is read. Existing
output directories are rejected. Only official training inputs and labels are
opened; no validation/test/audit inputs or reference answers are loaded.

These are fixed historical stability checks on development training labels,
not untouched holdouts. Final V8 selection followed a separately authorized
retrospective comparison; reproducing these checks does not alter that history.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='4'
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from catboost import CatBoostRanker, Pool
from common import DATA, check_data, load_inputs, read_jsonl
from student_features import prepare_inputs, group_correctness, choose_from_scores
from equation_features import prepare_equations
from tree_features import prepare_extra, feature_array
from ensemble_model import pool_scores
from v8_stack_features import prepare, subset, fit_projection, features
from v8_cat_numeric import export_numeric, score_features
from train_v8 import fit_base_models, sha, HEAD, BASE_SPEC, SOURCE_FILES, NAMES

CODE=Path(__file__).resolve().parent
CANDIDATE_SCALE=.2744070887565613
REFERENCE_SCALE=.8949364423751831


def _write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def _center(scores,groups):
    result=np.zeros_like(scores,dtype=np.float32)
    for row,answer_groups in enumerate(groups):
        valid=[index for members in answer_groups.values() for index in members]
        if valid:result[row,valid]=scores[row,valid]-scores[row,valid].mean()
    return result


def _reference(raw,groups):
    result=np.zeros((len(groups),4),np.float32)
    for name in NAMES:
        spec=BASE_SPEC[name];value=pool_scores(raw[name],groups,spec['pooling'])
        value-=value.mean(axis=1,keepdims=True)
        result+=spec['weight']*value/spec['oof_scale']
    return _center(result,groups)/REFERENCE_SCALE


def _candidate(probability,groups):
    return _center(pool_scores(probability,groups,'mean'),groups)/CANDIDATE_SCALE


def _hits(scores,groups,truth):
    choices=choose_from_scores(groups,scores,0.)
    return truth[np.arange(len(groups)),choices]


def _fit_head(x,truth,groups,query,projection,directory):
    """Fixed PairLogit fit, numeric export and unlabeled-query replay check."""
    started=time.perf_counter();pairs=[];weights=[]
    for row,answer_groups in enumerate(groups):
        valid=[index for members in answer_groups.values() for index in members]
        positive=[index for index in valid if truth[row,index]]
        negative=[index for index in valid if not truth[row,index]]
        if not positive or not negative:raise AssertionError('Hard group lacks a class')
        for good in positive:
            for bad in negative:
                pairs.append([row*4+good,row*4+bad]);weights.append(1/(len(positive)*len(negative)))
    pool=Pool(x.reshape(-1,x.shape[-1]),truth.ravel().astype(int),
              group_id=np.repeat(np.arange(len(truth)),4),pairs=pairs,pairs_weight=weights)
    model=CatBoostRanker(**HEAD).fit(pool)
    export_numeric(model,directory/'bundle');np.savez(directory/'bundle/projection.npz',**projection)
    expected=np.asarray(model.predict(query.reshape(-1,query.shape[-1]),thread_count=4)).reshape(query.shape[:2])
    actual=score_features(query,directory/'bundle')
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
    scores=(1/(1+np.exp(-np.clip(actual,-30,30)))).astype(np.float32)
    return scores,dict(seconds=time.perf_counter()-started,
                       native_numpy_max_error=float(np.max(abs(actual-expected))),
                       fit_questions=len(truth),feature_dim=x.shape[-1],pairs=len(pairs))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(20261031,20261101),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=DATA)
    args=parser.parse_args();output=args.output.resolve();data_dir=args.data.resolve()
    if output.exists():raise ValueError('Refusing to overwrite an existing output directory')
    output.mkdir(parents=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    started=time.perf_counter();check_data(data_dir,splits=['train'])
    bank,arrays=load_inputs('train',data_dir);references=read_jsonl(data_dir/'gsm8k_train.jsonl')
    if [(r['id'],r['question']) for r in bank]!=[(r['id'],r['question']) for r in references]:
        raise AssertionError('Training labels and candidate bank are misaligned')
    answers,groups,embedding,basic=prepare_inputs(bank,arrays)
    truth=group_correctness(answers,references)
    _,_,_,equation_text=prepare_equations(bank,arrays)
    _,_,_,tree_text=prepare_extra(bank,arrays)
    data=dict(groups=groups,truth=truth,ridge=np.concatenate([embedding,equation_text],axis=-1),
              neural=np.concatenate([embedding,basic],axis=-1),
              tree=feature_array(embedding,tree_text,groups,'raw'))
    stack_data=prepare(bank,arrays)
    hard=np.flatnonzero([len(group)>=2 for group in groups]);all_indices=np.arange(len(bank))
    inner_seed_base=20261110 if args.seed==20261031 else 20261120
    source_files=tuple(SOURCE_FILES)+('cross_validate_v8.py',)
    protocol=dict(recipe='fixed_v8_pair4',outer_seed=args.seed,outer_folds=5,
        inner_folds=3,inner_seed_base=inner_seed_base,head_config=HEAD,base_specification=BASE_SPEC,
        candidate_scale=CANDIDATE_SCALE,reference_scale=REFERENCE_SCALE,mixture_weight=1,
        source_sha256={name:sha(CODE/name) for name in source_files},
        train_sha256={name:sha(data_dir/name) for name in ('gsm8k_train.jsonl','candidates/gsm8k_train.jsonl','candidates/gsm8k_train.npz')},
        versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','torch','lightgbm','catboost')},
        threads=4,device='cpu',cache_reuse=False,validation_accessed=False,test_accessed=False,audit_accessed=False)
    _write(output/'protocol.json',protocol)
    base_oof={name:np.full((len(hard),4),np.nan,np.float32) for name in NAMES}
    head_oof=np.full((len(hard),4),np.nan,np.float32)
    folds=np.array_split(np.random.default_rng(args.seed).permutation(hard),5)
    records=[];outer_coverage=[]
    for fold,held in enumerate(folds):
        fold_started=time.perf_counter();folder=output/f'outer{fold}';folder.mkdir()
        outer_fit=np.setdiff1d(all_indices,held);fit_hard=np.setdiff1d(hard,held)
        positions=np.searchsorted(hard,held)
        if not np.array_equal(hard[positions],held):raise AssertionError('Outer row alignment differs')
        tick=time.perf_counter();query_raw=fit_base_models(data,outer_fit,query=held)
        outer_seconds=time.perf_counter()-tick
        for name in NAMES:base_oof[name][positions]=query_raw[name]
        # Float32 query scores match the historical saved outer base OOF.
        query_raw={name:base_oof[name][positions].copy() for name in NAMES}
        inner_seed=inner_seed_base+fold
        inner_folds=np.array_split(np.random.default_rng(inner_seed).permutation(fit_hard),3)
        inner_oof={name:np.full((len(fit_hard),4),np.nan,np.float32) for name in NAMES}
        inner_records=[];inner_coverage=[]
        for inner,inner_held in enumerate(inner_folds):
            fit=np.setdiff1d(outer_fit,inner_held)
            if (np.intersect1d(fit,held).size or np.intersect1d(fit,inner_held).size
                    or np.intersect1d(inner_held,held).size):raise AssertionError('Nested split overlap')
            tick=time.perf_counter();prediction=fit_base_models(data,fit,query=inner_held)
            inner_positions=np.searchsorted(fit_hard,inner_held)
            if not np.array_equal(fit_hard[inner_positions],inner_held):raise AssertionError('Inner row alignment differs')
            for name in NAMES:inner_oof[name][inner_positions]=prediction[name]
            inner_records.append(dict(inner_fold=inner,seed=inner_seed,fit_indices=fit.tolist(),
                                      held_indices=inner_held.tolist(),outer_excluded_indices=held.tolist(),
                                      seconds=time.perf_counter()-tick))
            inner_coverage.extend(inner_held.tolist())
            print(json.dumps(dict(stage='nested_base_oof',outer_fold=fold+1,inner_fold=inner+1,
                                  seconds=inner_records[-1]['seconds'])),flush=True)
        if sorted(inner_coverage)!=fit_hard.tolist() or len(set(inner_coverage))!=len(fit_hard):
            raise AssertionError('Inner OOF coverage differs')
        if not all(np.isfinite(s).all() for s in inner_oof.values()):raise AssertionError('Unfilled inner OOF')
        np.savez(folder/'inner_oof.npz',hard_indices=fit_hard,**inner_oof)
        projection=fit_projection(stack_data,fit_hard)
        fit_data,query_data=subset(stack_data,fit_hard),subset(stack_data,held)
        x=features(fit_data,inner_oof,projection,BASE_SPEC)
        qx=features(query_data,query_raw,projection,BASE_SPEC)
        probability,head_record=_fit_head(x,truth[fit_hard],fit_data['groups'],qx,projection,folder)
        head_oof[positions]=probability
        np.savez(folder/'held_scores.npz',held_indices=held,scores=probability,**query_raw)
        candidate_correct=int(_hits(_candidate(probability,query_data['groups']),query_data['groups'],truth[held]).sum())
        reference_correct=int(_hits(_reference(query_raw,query_data['groups']),query_data['groups'],truth[held]).sum())
        record=dict(fold=fold,outer_seed=args.seed,outer_fit_indices=outer_fit.tolist(),
            head_fit_indices=fit_hard.tolist(),outer_held_indices=held.tolist(),
            inner_records=inner_records,head=head_record,outer_base_seconds=outer_seconds,
            fold_seconds=time.perf_counter()-fold_started,candidate_correct=candidate_correct,
            reference_correct=reference_correct,net_gain=candidate_correct-reference_correct,
            inner_oof_sha256=sha(folder/'inner_oof.npz'),held_scores_sha256=sha(folder/'held_scores.npz'))
        _write(folder/'record.json',record);records.append(record);outer_coverage.extend(held.tolist())
        print(json.dumps(dict(stage='outer_complete',fold=fold+1,candidate_correct=candidate_correct,
                              reference_correct=reference_correct,seconds=record['fold_seconds'])),flush=True)
    if sorted(outer_coverage)!=hard.tolist() or len(set(outer_coverage))!=len(hard):raise AssertionError('Outer coverage differs')
    if not np.isfinite(head_oof).all() or not all(np.isfinite(s).all() for s in base_oof.values()):raise AssertionError('Unfilled outer OOF')
    np.savez(output/'base_oof.npz',hard_indices=hard,**base_oof)
    np.savez(output/'oof.npz',hard_indices=hard,scores=head_oof)
    hg=[groups[int(i)] for i in hard]
    candidate_hits=_hits(_candidate(head_oof,hg),hg,truth[hard])
    reference_hits=_hits(_reference(base_oof,hg),hg,truth[hard])
    candidate_correct=int(candidate_hits.sum());reference_correct=int(reference_hits.sum())
    deltas=[record['net_gain'] for record in records]
    if candidate_correct-reference_correct!=sum(deltas):raise AssertionError('Fold aggregate differs')
    expected=(1756,1741,[1,4,6,-1,5]) if args.seed==20261031 else (1752,1739,[-4,9,-1,-1,10])
    summary=dict(recipe='fixed_v8_pair4',outer_seed=args.seed,inner_seed_base=inner_seed_base,
        oof_correct=candidate_correct,oof_reference=reference_correct,oof_n=len(hard),
        net_gain=candidate_correct-reference_correct,fold_delta=deltas,
        fold_candidate_correct=[record['candidate_correct'] for record in records],
        fold_reference_correct=[record['reference_correct'] for record in records],
        matches_historical=(candidate_correct,reference_correct,deltas)==expected,
        historical_expected=dict(candidate_correct=expected[0],reference_correct=expected[1],fold_delta=expected[2]),
        positive_folds=sum(delta>0 for delta in deltas),base_triplets_fitted=20,
        inner_base_triplets=15,outer_base_triplets=5,heads_fitted=5,
        elapsed_seconds=time.perf_counter()-started,
        base_fit_seconds=sum(record['outer_base_seconds']+sum(inner['seconds'] for inner in record['inner_records']) for record in records),
        head_fit_seconds=sum(record['head']['seconds'] for record in records),
        maximum_native_numpy_error=max(record['head']['native_numpy_max_error'] for record in records),
        base_oof_sha256=sha(output/'base_oof.npz'),oof_sha256=sha(output/'oof.npz'),
        protocol_sha256=sha(output/'protocol.json'),
        fold_record_sha256={f'outer{fold}/record.json':sha(output/f'outer{fold}/record.json') for fold in range(5)},
        threads=4,device='cpu',new_parameter_selection=False,cache_reuse=False,
        validation_accessed=False,test_accessed=False,audit_accessed=False,
        caveat='Historical development labels are reused. This is a fixed-recipe reproduction, not an untouched holdout or test result.')
    if args.seed==20261101:
        summary['second_seed_gate_passed']=summary['net_gain']>=5 and summary['positive_folds']>=3
    _write(output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
