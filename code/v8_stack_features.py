"""Independent, input-only features for nested stacking experiments.

The caller must supply genuinely out-of-fold base scores when fitting a
stacker. This feature module does not create folds, read labels, fit base
models, or inspect IDs. Only fit_projection imports a training dependency;
prepare, subset, and features use the standard library and NumPy.
"""

import numpy as np

from tree_features import prepare_extra
from trajectory_features_v6 import prepare_trajectory


BASE_NAMES = ('equations', 'neural', 'tree')
PCA_DIM = 32
TEXT_DIM = 127
SCORE_NAMES = (
    'raw_scaled', 'centered_scaled', 'configured_pool_scaled',
    'configured_pool_centered', 'group_mean_scaled', 'group_max_scaled',
    'group_min_scaled', 'group_std_scaled', 'margin_to_best_other_group',
    'sigmoid_centered', 'candidate_average_rank', 'group_average_rank',
)
GROUP_NAMES = ('valid', 'own_group_fraction', 'distinct_group_fraction',
               'valid_candidate_fraction', 'largest_group_fraction',
               'own_minus_largest_fraction')
FEATURE_NAMES = (
    tuple(f'text_{i:03d}' for i in range(TEXT_DIM))
    + tuple(f'text_difference_{i:03d}' for i in range(TEXT_DIM))
    + tuple(f'pca_{kind}_{i:02d}' for kind in ('raw', 'difference', 'context') for i in range(PCA_DIM))
    + tuple(f'{base}_{name}' for base in BASE_NAMES for name in SCORE_NAMES)
    + GROUP_NAMES
)
FEATURE_DIM = len(FEATURE_NAMES)
assert FEATURE_DIM == 392


def _validate(data):
    groups = data['groups']
    count = len(groups)
    embeddings = np.asarray(data['embeddings'])
    text = np.asarray(data['text'])
    if embeddings.shape != (count, 4, 1024) or text.shape != (count, 4, TEXT_DIM):
        raise ValueError('Expected embeddings [N,4,1024] and text [N,4,127]')
    if not np.issubdtype(embeddings.dtype, np.number) or not np.issubdtype(text.dtype, np.number):
        raise ValueError('Feature arrays must be numeric')
    if not np.isfinite(embeddings).all() or not np.isfinite(text).all():
        raise ValueError('Input features contain non-finite values')
    valid = np.zeros((count, 4), dtype=bool)
    for row, answer_groups in enumerate(groups):
        if not isinstance(answer_groups, dict):
            raise ValueError('Each row must contain a parsed-answer group dictionary')
        for answer, indices in answer_groups.items():
            if answer is None or not len(indices):
                raise ValueError('Answer groups must be nonempty and parseable')
            for index in indices:
                if not isinstance(index, (int, np.integer)) or index < 0 or index >= 4 or valid[row, index]:
                    raise ValueError('Group members must be distinct candidate indices in 0..3')
                valid[row, index] = True
    return embeddings, text, valid


def prepare(bank, arrays):
    """Combine the existing 107 diagnostics with 20 trajectory diagnostics."""
    if not len(bank):
        if np.asarray(arrays['embeddings']).shape != (0, 4, 1024):
            raise ValueError('Unexpected empty embedding shape')
        return dict(groups=[], embeddings=np.empty((0, 4, 1024), np.float32),
                    text=np.empty((0, 4, TEXT_DIM), np.float32))
    answers, groups, embeddings, original = prepare_extra(bank, arrays)
    trajectory_answers, trajectory_groups, trajectory_embeddings, trajectory = prepare_trajectory(bank, arrays)
    if answers != trajectory_answers or groups != trajectory_groups:
        raise AssertionError('Existing feature modules disagree about parsed answer groups')
    if original.shape != (len(bank), 4, 107) or trajectory.shape != (len(bank), 4, 62):
        raise AssertionError('Existing diagnostic dimensions changed')
    if not np.array_equal(embeddings, trajectory_embeddings):
        raise AssertionError('Existing feature modules changed embedding alignment')
    result = dict(groups=groups, embeddings=np.asarray(embeddings, np.float32),
                  text=np.concatenate((original, trajectory[:, :, -20:]), axis=-1).astype(np.float32))
    _validate(result)
    return result


def subset(data, indices):
    """Take aligned question rows, preserving groups and numeric inputs."""
    embeddings, text, _ = _validate(data)
    indices = np.asarray(indices)
    if indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer)):
        raise ValueError('Subset indices must be a one-dimensional integer sequence')
    indices = indices.astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= len(data['groups'])):
        raise ValueError('Subset indices are outside the supplied data')
    return dict(groups=[{answer: list(members) for answer, members in data['groups'][int(i)].items()} for i in indices],
                embeddings=embeddings[indices], text=text[indices])


def fit_projection(data, indices):
    """Fit PCA exclusively on the supplied question indices; return NumPy assets."""
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    indices = np.asarray(indices)
    if not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError('Projection fit indices must be nonempty and unique')
    fitted = subset(data, indices)
    raw = fitted['embeddings'].reshape(-1, 1024)
    rank = min(PCA_DIM, len(raw) - 1, 1024)
    with threadpool_limits(limits=4):
        pca = PCA(n_components=rank, svd_solver='randomized', random_state=20261104, iterated_power=3)
        pca.fit(raw)
    components = np.zeros((PCA_DIM, 1024), dtype=np.float32)
    components[:rank] = pca.components_.astype(np.float32)
    assets = dict(mean=pca.mean_.astype(np.float32), components=components)
    if not all(np.isfinite(value).all() for value in assets.values()):
        raise ValueError('Projection fit produced non-finite assets')
    return assets


def _average_rank(values):
    """Ascending [0,1] ranks, with ties at their shared average position."""
    values = np.asarray(values)
    if len(values) == 1:
        return np.array([.5], dtype=np.float64)
    if not len(values):
        return np.empty(0, dtype=np.float64)
    lesser = (values[:, None] > values[None, :]).sum(axis=1)
    equal_other = (values[:, None] == values[None, :]).sum(axis=1) - 1
    return (lesser + .5 * equal_other) / (len(values) - 1)


def _score_features(raw, groups, valid, member):
    scale = float(member['oof_scale'])
    pooling = member['pooling']
    if not np.isfinite(scale) or scale <= 0 or pooling not in ('max', 'mean'):
        raise ValueError('Each base requires positive finite oof_scale and max/mean pooling')
    scaled = np.asarray(raw, dtype=np.float64) / scale
    if scaled.shape != valid.shape or not np.isfinite(scaled).all():
        raise ValueError('Each base score array must be finite and have shape [N,4]')
    result = np.zeros((*valid.shape, len(SCORE_NAMES)), dtype=np.float64)
    reduce = np.max if pooling == 'max' else np.mean
    for row, answer_groups in enumerate(groups):
        indices = np.flatnonzero(valid[row])
        if not len(indices):
            continue
        values = scaled[row, indices]
        centered = values - values.mean(dtype=np.float64)
        result[row, indices, 0] = values
        result[row, indices, 1] = centered
        result[row, indices, 9] = 1 / (1 + np.exp(-np.clip(centered, -30, 30)))
        result[row, indices, 10] = _average_rank(values)
        member_lists = list(answer_groups.values())
        grouped = np.asarray([reduce(scaled[row, members]) for members in member_lists])
        # Each distinct answer contributes once to the group baseline/rank.
        group_center = grouped.mean(dtype=np.float64)
        group_ranks = _average_rank(grouped)
        for group_index, members in enumerate(member_lists):
            own = scaled[row, members]
            others = np.delete(grouped, group_index)
            result[row, members, 2] = grouped[group_index]
            result[row, members, 3] = grouped[group_index] - group_center
            result[row, members, 4] = own.mean(dtype=np.float64)
            result[row, members, 5] = own.max()
            result[row, members, 6] = own.min()
            result[row, members, 7] = own.std(dtype=np.float64)
            result[row, members, 8] = grouped[group_index] - others.max() if len(others) else 0.
            result[row, members, 11] = group_ranks[group_index]
    return result


def features(data, raw_scores_dict, projection, spec):
    """Build fixed 392-column features without reading labels or fitting state.

    raw_scores_dict has exactly the three base names, each aligned [N,4]. spec
    may be their member dictionary or an ensemble dictionary with 'members'.
    The caller is responsible for fitting the projection and supplying base
    scores/scales using only the appropriate training side of each fold.
    """
    embedding, text, valid = _validate(data)
    if set(raw_scores_dict) != set(BASE_NAMES):
        raise ValueError('Expected equations, neural, and tree raw score arrays')
    mean = np.asarray(projection['mean'])
    components = np.asarray(projection['components'])
    if mean.shape != (1024,) or components.shape != (PCA_DIM, 1024):
        raise ValueError('Projection requires mean [1024] and components [32,1024]')
    if not np.isfinite(mean).all() or not np.isfinite(components).all():
        raise ValueError('Projection contains non-finite values')
    projected = (embedding - mean) @ components.T
    projected = projected.astype(np.float64)
    context = projected.mean(axis=1, keepdims=True, dtype=np.float64)
    diagnostics = text.astype(np.float64)
    columns = [diagnostics, diagnostics - diagnostics.mean(axis=1, keepdims=True),
               projected, projected - context, np.broadcast_to(context, projected.shape)]
    members = spec['members'] if 'members' in spec else spec
    for name in BASE_NAMES:
        columns.append(_score_features(raw_scores_dict[name], data['groups'], valid, members[name]))
    group_features = np.zeros((*valid.shape, len(GROUP_NAMES)), dtype=np.float64)
    for row, groups in enumerate(data['groups']):
        group_features[row, :, 0] = valid[row]
        group_features[row, :, 2] = len(groups) / 4
        group_features[row, :, 3] = valid[row].sum() / 4
        largest = max((len(indices) for indices in groups.values()), default=0)
        group_features[row, :, 4] = largest / 4
        for indices in groups.values():
            group_features[row, indices, 1] = len(indices) / 4
            group_features[row, indices, 5] = (len(indices) - largest) / 4
    columns.append(group_features)
    output = np.concatenate(columns, axis=-1)
    if output.shape != (len(data['groups']), 4, FEATURE_DIM) or not np.isfinite(output).all():
        raise ValueError('Stack features have invalid dimensions or non-finite entries')
    output = output.astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError('Stack features overflow float32')
    return output


def _sanity():
    """Fabricated examples only; no fit_projection, model fitting, or data IO."""
    import copy
    import sys
    from itertools import permutations

    rng = np.random.default_rng(4)
    text_rows = [
        ['2+3=5; #### 5', '5 is the result. #### 5', '2+3=6; #### 6', 'No final answer'],
        ['2*4=8; #### 8'] * 4,
        ['No numeric final answer'] * 4,
    ]
    bank = [dict(id=f'ignored-{row}', question='A fabricated arithmetic comparison.',
                 candidates=[dict(index=i, text=text) for i, text in enumerate(texts)])
            for row, texts in enumerate(text_rows)]
    arrays = dict(embeddings=rng.normal(size=(3, 4, 1024)).astype(np.float32))
    data = prepare(bank, arrays)
    projection = dict(mean=np.zeros(1024, np.float32), components=np.eye(PCA_DIM, 1024, dtype=np.float32))
    raw = {name: np.array([[.2, .2, -.3, -1e6], [1., 2., 3., 4.], [-1e6]*4], np.float32)
           for name in BASE_NAMES}
    spec = {name: dict(pooling='mean' if name == 'tree' else 'max', oof_scale=.7)
            for name in BASE_NAMES}
    reference = features(data, raw, projection, spec)
    assert reference.shape == (3, 4, FEATURE_DIM) and FEATURE_DIM == 392
    assert np.isfinite(reference).all()
    score_start = 2 * TEXT_DIM + 3 * PCA_DIM
    assert np.all(reference[2, :, score_start:score_start + 3*len(SCORE_NAMES)] == 0)
    ranks = _average_rank(np.array([2., 2., 1., 4.]))
    np.testing.assert_allclose(ranks, [.5, .5, 0, 1])
    assert reference[0, 0, score_start + 10] == reference[0, 1, score_start + 10]
    for order in permutations(range(4)):
        perm = np.asarray(order)
        permuted = copy.deepcopy(bank)
        for row in permuted:
            row['id'] = 'arbitrary replacement ID'
            row['candidates'] = [row['candidates'][i] for i in perm]
            for i, candidate in enumerate(row['candidates']):
                candidate['index'] = 80 - i
        changed = prepare(permuted, dict(embeddings=arrays['embeddings'][:, perm]))
        output = features(changed, {name: value[:, perm] for name, value in raw.items()}, projection, spec)
        np.testing.assert_allclose(output, reference[:, perm], atol=2e-6, rtol=2e-6)
    order = np.array([2, 0, 1])
    np.testing.assert_allclose(features(subset(data, order), {name: value[order] for name, value in raw.items()}, projection, {'members': spec}), reference[order])
    assert features(subset(data, []), {name: value[:0] for name, value in raw.items()}, projection, spec).shape == (0, 4, FEATURE_DIM)
    bad = {name: value.copy() for name, value in raw.items()}
    bad['tree'][0, 0] = np.nan
    try:
        features(data, bad, projection, spec)
    except ValueError:
        pass
    else:
        raise AssertionError('NaN score was accepted')
    assert 'sklearn' not in sys.modules and 'scipy' not in sys.modules
    print('v8_stack_features sanity passed: 392 columns; 24 permutations; arbitrary IDs; tied ranks; single/no-answer groups; row/empty subset; NaN rejection; NumPy inference.')


if __name__ == '__main__':
    _sanity()
