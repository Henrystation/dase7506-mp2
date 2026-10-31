"""Scalar numeric symmetric CatBoost trees, exported to NumPy-only inference.

Format source: https://catboost.ai/docs/en/features/export-model-to-json
Only FloatFeature splits and scalar oblivious-tree leaves are supported.
Categorical/CTR, text/embedding processing, asymmetric trees and vector-valued
models are rejected. Training-library imports occur only in the fabricated
self-check, never in inference or the exporter.
"""
import json
from pathlib import Path
import tempfile

import numpy as np

VERSION = 'mp2_catboost_numeric_symmetric_v8'
SOURCE = 'https://catboost.ai/docs/en/features/export-model-to-json'


def _numeric_arrays(document, expected_features=None):
    if 'trees' in document or 'oblivious_trees' not in document:
        raise ValueError('Only symmetric oblivious trees are supported')
    info = document.get('features_info', {})
    for name in ('categorical_features', 'ctrs', 'text_features', 'embedding_features',
                 'estimated_features'):
        if info.get(name):
            raise ValueError(f'Unsupported nonnumeric feature metadata: {name}')
    if document.get('ctr_data'):
        raise ValueError('CTR tables are unsupported')
    mapping, nan_right = {}, {}
    for feature in info.get('float_features', []):
        index, flat = int(feature['feature_index']), int(feature['flat_feature_index'])
        if index < 0 or flat < 0 or index in mapping:
            raise ValueError('Invalid float-feature index mapping')
        mapping[index] = flat
        treatment = feature.get('nan_value_treatment', 'AsIs')
        if treatment not in ('AsIs', 'AsFalse', 'AsTrue'):
            raise ValueError(f'Unsupported NaN treatment: {treatment}')
        nan_right[index] = treatment == 'AsTrue'
    minimum_features = max(mapping.values(), default=-1) + 1
    feature_count = minimum_features if expected_features is None else int(expected_features)
    if feature_count < minimum_features or feature_count < 1:
        raise ValueError('Invalid model feature count')
    features, borders, nans, leaves = [], [], [], []
    split_offsets, leaf_offsets = [0], [0]
    for tree in document['oblivious_trees']:
        splits = tree.get('splits') or []
        # An int64 leaf index is used at runtime. Huge trees are not suitable
        # for this compact deployment format; actual experiments use depth 4/6.
        if len(splits) > 20:
            raise ValueError('Unsupported tree depth greater than 20')
        values = np.asarray(tree['leaf_values'], dtype=np.float64)
        if values.ndim != 1 or len(values) != 2**len(splits):
            raise ValueError('Only one scalar output per leaf is supported')
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite leaf values')
        for split in splits:
            if split.get('split_type') != 'FloatFeature':
                raise ValueError('Only FloatFeature splits are supported')
            fi = int(split['float_feature_index'])
            if fi not in mapping:
                raise ValueError('Split references missing float-feature metadata')
            border = float(split['border'])
            if not np.isfinite(border):
                raise ValueError('Nonfinite split border')
            features.append(mapping[fi]); borders.append(border); nans.append(nan_right[fi])
        leaves.extend(values.tolist())
        split_offsets.append(len(features)); leaf_offsets.append(len(leaves))
    scale_and_bias = document.get('scale_and_bias', [1., [0.]])
    if (len(scale_and_bias) != 2 or not isinstance(scale_and_bias[1], list)
            or len(scale_and_bias[1]) != 1):
        raise ValueError('Only scalar scale-and-bias is supported')
    scale, bias = float(scale_and_bias[0]), float(scale_and_bias[1][0])
    if not np.isfinite([scale, bias]).all():
        raise ValueError('Nonfinite scale or bias')
    arrays = dict(feature=np.asarray(features, np.int32), border=np.asarray(borders, np.float64),
                  nan_right=np.asarray(nans, bool), leaf_values=np.asarray(leaves, np.float64),
                  split_offsets=np.asarray(split_offsets, np.int64),
                  leaf_offsets=np.asarray(leaf_offsets, np.int64),
                  scale=np.asarray(scale, np.float64), bias=np.asarray(bias, np.float64))
    return arrays, feature_count


def export_numeric(model, output_dir):
    """Export a fitted scalar numeric CatBoost model; return asset metadata."""
    # CatBoost's own exporter supplies its documented JSON representation. The
    # full JSON is temporary and is not retained among inference assets.
    with tempfile.TemporaryDirectory(prefix='mp2-catboost-export-') as temp:
        path = Path(temp)/'model.json'
        model.save_model(str(path), format='json')
        document = json.loads(path.read_text(encoding='utf8'))
    n_features = getattr(model, 'n_features_in_', None)
    if n_features is None:
        names = getattr(model, 'feature_names_', None)
        n_features = len(names) if names else None
    arrays, feature_count = _numeric_arrays(document, n_features)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    weight_file = output_dir/'cat_weights.npz'
    config_file = output_dir/'cat_config.json'
    np.savez(weight_file, **arrays)
    record = dict(version=VERSION, n_features=feature_count,
                  num_trees=len(arrays['split_offsets'])-1,
                  num_splits=len(arrays['feature']), num_leaves=len(arrays['leaf_values']),
                  max_depth=int(np.diff(arrays['split_offsets']).max(initial=0)),
                  array_bytes=sum(a.nbytes for a in arrays.values()),
                  weights_file_bytes=weight_file.stat().st_size,
                  output='RawFormulaVal', input_cast='float32', numeric_dtype='float64',
                  source=SOURCE)
    config_file.write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
    return {**record, 'asset_file_bytes': weight_file.stat().st_size+config_file.stat().st_size,
            'config_file_bytes': config_file.stat().st_size}


def score_features(features_anyNd, checkpoint):
    """Return raw scalar logits with shape features.shape[:-1], including N=0.

    Numeric input is quantized to float32 before comparisons, matching native
    CatBoost input handling. Borders/leaves/scale/bias stay float64. There is no
    sigmoid or answer pooling in this low-level tree evaluator.
    """
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint/'cat_config.json').read_text(encoding='utf8'))
    if config.get('version') != VERSION:
        raise ValueError('Unsupported numeric CatBoost checkpoint')
    x = np.asarray(features_anyNd)
    if x.ndim < 1 or x.shape[-1] != config['n_features']:
        raise ValueError('Feature dimension differs from the checkpoint')
    if not (np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.bool_)) or np.iscomplexobj(x):
        raise ValueError('Expected real numeric features')
    shape = x.shape[:-1]
    x = np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
    with np.load(checkpoint/'cat_weights.npz', allow_pickle=False) as p:
        f, border, nan = p['feature'], p['border'], p['nan_right']
        splits, offsets, leaves = p['split_offsets'], p['leaf_offsets'], p['leaf_values']
        total = np.zeros(len(x), np.float64)
        for tree in range(len(splits)-1):
            first, last = int(splits[tree]), int(splits[tree+1])
            index = np.zeros(len(x), np.int64)
            for bit, split in enumerate(range(first, last)):
                values = x[:, f[split]]
                right = values > border[split]
                if nan[split]:
                    right |= np.isnan(values)
                index |= right.astype(np.int64) << bit
            total += leaves[int(offsets[tree])+index]
        return (total*float(p['scale'])+float(p['bias'])).reshape(shape)


def _fabricated_export_check():
    """Small generated arrays only: classifier/ranker precision and boundaries."""
    from catboost import CatBoostClassifier, CatBoostRanker
    rng = np.random.default_rng(20261105)
    x = rng.normal(size=(160, 5)).astype(np.float32)
    x[:, 1] = 3.  # Exercises the flat-feature mapping past an unused feature.
    y = (x[:,0] + .4*x[:,3] - .25*x[:,4] > 0).astype(int)
    group = np.repeat(np.arange(40), 4)
    ranking_y = np.zeros((40,4), np.int32)
    ranking_y[np.arange(40), np.argmax((x[:,0]+.3*x[:,2]).reshape(40,4),axis=1)] = 1
    options = dict(iterations=19, depth=3, learning_rate=.08, random_seed=17,
                   thread_count=4, verbose=False, allow_writing_files=False)
    models = [('classifier', CatBoostClassifier(loss_function='Logloss', **options).fit(x,y)),
              ('ranker', CatBoostRanker(loss_function='PairLogit', **options).fit(x,ranking_y.ravel(),group_id=group))]
    nan_x=x.copy();nan_x[::7,0]=np.nan
    models.extend((f'nan_{mode}',CatBoostClassifier(loss_function='Logloss',nan_mode=mode,**options).fit(nan_x,y))
                  for mode in ('Min','Max'))
    records = []
    with tempfile.TemporaryDirectory(prefix='mp2-catboost-sanity-') as temp:
        for name, model in models:
            # Nonidentity scale/bias must also be preserved by the export.
            model.set_scale_and_bias(1.7, -.23)
            path = Path(temp)/name
            record = export_numeric(model,path)
            samples = rng.normal(size=(7,5))
            with np.load(path/'cat_weights.npz') as weights:
                boundary=[]
                for fi, threshold in zip(weights['feature'][:12],weights['border'][:12]):
                    value = np.float32(threshold)
                    for probe in (np.nextafter(value,np.float32(-np.inf)),value,
                                  np.nextafter(value,np.float32(np.inf))):
                        row=np.zeros(5,np.float64);row[fi]=probe;boundary.append(row)
                    # A float64 step below a float32 border rounds back to it.
                    row=np.zeros(5,np.float64);row[fi]=np.nextafter(float(value),-np.inf);boundary.append(row)
                samples=np.concatenate([samples,boundary])
            nan_rows=np.zeros((3,5));nan_rows[:,0]=[np.nan,-np.inf,np.inf]
            samples=np.concatenate([samples,nan_rows])
            expected=(model.predict(samples,thread_count=4) if name=='ranker'
                      else model.predict(samples,prediction_type='RawFormulaVal',thread_count=4))
            actual=score_features(samples,path)
            maximum=float(np.max(np.abs(expected-actual)))
            np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
            np.testing.assert_allclose(score_features(samples[:1],path),expected[:1],atol=1e-7,rtol=0)
            np.testing.assert_allclose(score_features(samples[:7],path),expected[:7],atol=1e-7,rtol=0)
            np.testing.assert_allclose(score_features(samples[0],path),expected[0],atol=1e-7,rtol=0)
            np.testing.assert_allclose(score_features(samples[:6].reshape(2,3,5),path),expected[:6].reshape(2,3),atol=1e-7,rtol=0)
            assert score_features(np.empty((0,5)),path).shape==(0,)
            assert score_features(np.empty((2,0,5)),path).shape==(2,0)
            records.append(dict(kind=name,max_absolute_error=maximum,**record))
    print(json.dumps(dict(status='passed',synthetic_only=True,records=records),indent=2))


if __name__ == '__main__':
    _fabricated_export_check()
