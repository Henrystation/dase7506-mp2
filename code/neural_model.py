"""NumPy-only inference for the fixed five-epoch candidate verifier.

Only supplied inputs and learned checkpoint arrays are read here. Training and
the independent reference-answer scorer live in separate files.
"""
import json
from pathlib import Path

import numpy as np

from student_features import prepare_inputs
from student_model import score_candidates


def _gelu(values):
    """Match the tanh-approximation GELU used during PyTorch training."""
    return 0.5 * values * (
        1 + np.tanh(np.sqrt(2 / np.pi) * (values + 0.044715 * values ** 3))
    )


def candidate_scores(bank, arrays, checkpoint):
    """Return four scores per input question, parsed-answer groups and metadata."""
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    if config["mode"] not in ("raw", "diff", "both"):
        raise ValueError("Unknown neural feature mode")
    _, groups, embeddings, text_features = prepare_inputs(bank, arrays)
    features = np.concatenate([embeddings, text_features], axis=-1)
    with np.load(checkpoint / "weights.npz", allow_pickle=False) as params:
        raw = np.clip(
            (features - params["raw_mean"]) / params["raw_scale"], -8, 8
        )
        difference = np.clip(
            (features - features.mean(axis=1, keepdims=True)) / params["diff_scale"],
            -8, 8,
        )
        if config["mode"] == "raw":
            inputs = raw
        elif config["mode"] == "diff":
            inputs = difference
        else:
            inputs = np.concatenate([raw, difference], axis=-1)
        base = config["base_scale"] * score_candidates(
            features, params["ridge_weights"], params["ridge_scale"]
        )
        hidden = _gelu(inputs @ params["fc1.weight"].T + params["fc1.bias"])
        hidden = _gelu(hidden @ params["fc2.weight"].T + params["fc2.bias"])
        scores = base + (
            hidden @ params["out.weight"].T + params["out.bias"]
        ).squeeze(-1)
    if scores.shape != (len(bank), 4) or not np.isfinite(scores).all():
        raise ValueError("Invalid neural checkpoint output")
    return scores, groups, metadata
