"""Frozen offline answer-group ensemble for arbitrary MP2 input banks."""
import json
from pathlib import Path
import numpy as np
from student_features import choose_from_scores
from student_model import score_candidates
from equation_features import prepare_equations
from neural_model import candidate_scores as neural_scores
from tree_model import tree_scores


def pool_scores(scores, groups, aggregation):
    """Give each candidate the utility of its parsed-answer group."""
    pooled = np.asarray(scores, dtype=np.float32).copy()
    reduce = np.max if aggregation == 'max' else np.mean
    for i, answers in enumerate(groups):
        for indices in answers.values():
            pooled[i, indices] = reduce(scores[i, indices])
    return pooled


def answer_scores(bank, arrays, checkpoint):
    """Read only the supplied inputs and small frozen model assets."""
    if checkpoint is None:
        raise ValueError('Pass --checkpoint with the complete v2 bundle')
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'ensemble.json').read_text(encoding='utf8'))
    if config.get('version') != 'mp2_answer_group_ensemble_v2':
        raise ValueError('Unsupported ensemble configuration')
    _, groups, embeddings, text = prepare_equations(bank, arrays)
    features = np.concatenate([embeddings, text], axis=-1)
    with np.load(checkpoint / 'equations' / 'weights.npz', allow_pickle=False) as p:
        ridge = score_candidates(features, p['weights'], p['scale'])
    neural, _, _ = neural_scores(bank, arrays, checkpoint / 'neural')
    tree = tree_scores(bank, arrays, checkpoint / 'tree')
    values = {'equations': ridge, 'neural': neural, 'tree': tree}
    combined = np.zeros((len(bank), 4), dtype=np.float32)
    for name, spec in config['members'].items():
        scores = pool_scores(values[name], groups, spec['pooling'])
        scores -= scores.mean(axis=1, keepdims=True)
        combined += spec['weight'] * scores / spec['oof_scale']
    return combined, groups, config


def predict(bank, arrays, checkpoint):
    scores, groups, config = answer_scores(bank, arrays, checkpoint)
    return choose_from_scores(groups, scores, config['vote_strength'], 'max')
