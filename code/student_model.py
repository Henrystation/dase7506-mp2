"""Small, entirely local verifier trained on the supplied MP2 training split."""

import json
from pathlib import Path

import numpy as np

from student_features import choose_from_scores, prepare_inputs


MODEL_VERSION = "group_centered_ridge_v1"


def candidate_features(embeddings, text_features, feature_set):
    if feature_set == "embedding":
        return embeddings
    if feature_set == "text":
        return text_features
    if feature_set == "both":
        return np.concatenate((embeddings, text_features), axis=-1)
    raise ValueError(f"Unknown feature set: {feature_set}")


def fit_ridge(features, correctness, groups, regularization):
    """Fit within-question ridge from training examples only.

    Centering cancels question-specific information and turns the problem into
    comparing the four candidates, rather than memorizing the question bank.
    Only groups with at least two valid parsed answers have a choice to learn.
    """
    if regularization <= 0:
        raise ValueError("Regularization must be positive")
    hard = np.asarray([len(row) >= 2 for row in groups], dtype=bool)
    x = np.asarray(features[hard], dtype=np.float32)
    y = np.asarray(correctness[hard], dtype=np.float32)
    x -= x.mean(axis=1, keepdims=True)
    y -= y.mean(axis=1, keepdims=True)
    scale = np.maximum(x.std(axis=(0, 1)), 0.05)
    x = (x / scale).reshape(-1, x.shape[-1])
    y = y.ravel()
    count = x.shape[0]
    gram = (x.T @ x) / count
    gram.flat[:: gram.shape[0] + 1] += regularization
    target = (x.T @ y) / count
    weights = np.linalg.solve(gram, target).astype(np.float32)
    return weights, scale.astype(np.float32), int(hard.sum())


def score_candidates(features, weights, scale):
    centered = features - features.mean(axis=1, keepdims=True)
    return np.einsum("ncf,f->nc", centered / scale, weights, optimize=True)


def save_model(checkpoint, weights, scale, feature_set, regularization,
               vote_strength, aggregation, training_hard_groups):
    checkpoint = Path(checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint / "weights.npz", weights=weights, scale=scale)
    config = {
        "version": MODEL_VERSION,
        "feature_set": feature_set,
        "regularization": regularization,
        "vote_strength": vote_strength,
        "aggregation": aggregation,
        "training_hard_groups": training_hard_groups,
        "seed": 7506,
    }
    (checkpoint / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_model(checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("version") != MODEL_VERSION:
        raise ValueError("Unsupported model version")
    with np.load(checkpoint / "weights.npz", allow_pickle=False) as package:
        weights = package["weights"].copy()
        scale = package["scale"].copy()
    if (len(weights) != len(scale) or not np.isfinite(weights).all()
            or not np.isfinite(scale).all() or not np.all(scale > 0)):
        raise ValueError("Invalid checkpoint")
    return config, weights, scale


def predict_with_model(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("This method requires --checkpoint with config.json and weights.npz")
    config, weights, scale = load_model(checkpoint)
    _, groups, embeddings, text_features = prepare_inputs(bank, arrays)
    features = candidate_features(embeddings, text_features, config["feature_set"])
    scores = score_candidates(features, weights, scale)
    return choose_from_scores(groups, scores, config["vote_strength"], config["aggregation"])
