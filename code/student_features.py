"""Input-only candidate features for the MP2 verifier.

This module deliberately never opens reference answers. The official numeric
parser is imported unchanged from ``common`` so masking matches the scorer.
"""

from collections import Counter
import math
import re

import numpy as np

from common import parse_answer


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
WORD_RE = re.compile(r"[a-z]+")


def _numbers(text):
    return {item.replace(",", "") for item in NUMBER_RE.findall(text)}


def _text_features(question, text, answer):
    low = text.lower()
    qlow = question.lower()
    numbers = NUMBER_RE.findall(text)
    qnumbers = _numbers(question)
    cnumbers = _numbers(text)
    words = WORD_RE.findall(low)
    qwords = set(WORD_RE.findall(qlow))
    overlap = len(set(words) & qwords) / max(1, len(qwords))
    number_coverage = len(cnumbers & qnumbers) / max(1, len(qnumbers))
    answer_digits = len(answer) if answer is not None else 0
    return [
        math.log1p(len(text)),
        math.log1p(len(words)),
        math.log1p(len(numbers)),
        math.log1p(len(cnumbers)),
        math.log1p(text.count("\n")),
        math.log1p(text.count("=")),
        math.log1p(text.count("+")),
        math.log1p(text.count("-")),
        math.log1p(text.count("*")),
        math.log1p(text.count("/")),
        math.log1p(text.count("$")),
        math.log1p(text.count("\\frac")),
        math.log1p(text.count("\\boxed")),
        float("####" in text),
        float("final answer" in low),
        float("answer:" in low),
        float("therefore" in low),
        float("hence" in low),
        float("check" in low),
        float("verify" in low),
        float("wait" in low),
        float("however" in low),
        float("step" in low),
        float("%" in text),
        float("$" in text),
        overlap,
        number_coverage,
        math.log1p(answer_digits),
        float(answer is not None and "/" in answer),
        float(answer is not None and answer.startswith("-")),
        float(answer == "0"),
        float(answer == "1"),
    ]


def prepare_inputs(bank, arrays):
    """Return candidate texts, valid-answer groups, and numeric feature arrays."""
    embeddings = arrays["embeddings"]
    if embeddings.ndim != 3 or embeddings.shape[0] != len(bank) or embeddings.shape[1:] != (4, 1024):
        raise ValueError("Unexpected embedding shape")
    answers = []
    groups = []
    text_features = []
    for row in bank:
        if len(row["candidates"]) != 4:
            raise ValueError("Expected four candidates")
        row_answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        answer_groups = {}
        for index, answer in enumerate(row_answers):
            if answer is not None:
                answer_groups.setdefault(answer, []).append(index)
        answers.append(row_answers)
        groups.append(answer_groups)
        for candidate, answer in zip(row["candidates"], row_answers):
            text_features.append(_text_features(row["question"], candidate["text"], answer))
    return (
        answers,
        groups,
        np.asarray(embeddings, dtype=np.float32),
        np.asarray(text_features, dtype=np.float32).reshape(len(bank), 4, -1),
    )


def normalizer(embeddings, text_features):
    """Compute statistics from training inputs only."""
    flat_e = embeddings.reshape(-1, embeddings.shape[-1])
    flat_t = text_features.reshape(-1, text_features.shape[-1])
    e_mean = flat_e.mean(axis=0)
    e_std = np.maximum(flat_e.std(axis=0), 0.05)
    t_mean = flat_t.mean(axis=0)
    t_std = np.maximum(flat_t.std(axis=0), 0.1)
    return e_mean, e_std, t_mean, t_std


def normalized_features(embeddings, text_features, stats, use_embeddings=True, use_text=True):
    e_mean, e_std, t_mean, t_std = stats
    columns = []
    if use_embeddings:
        columns.append(np.clip((embeddings - e_mean) / e_std, -8, 8))
    if use_text:
        columns.append(np.clip((text_features - t_mean) / t_std, -8, 8))
    return np.concatenate(columns, axis=-1).astype(np.float32)


def choose_from_scores(groups, candidate_scores, vote_strength, aggregation="max"):
    """Choose a candidate from the most credible final-answer group."""
    result = []
    for answer_groups, row_scores in zip(groups, candidate_scores):
        if not answer_groups:
            # The official scorer regards every unparseable answer as wrong.
            result.append(0)
            continue
        if aggregation not in ("max", "mean"):
            raise ValueError("Unknown score aggregation")
        best_answer = max(
            answer_groups,
            key=lambda answer: (
                float((np.max if aggregation == "max" else np.mean)(row_scores[answer_groups[answer]]))
                + vote_strength * (len(answer_groups[answer]) - 1),
                len(answer_groups[answer]),
                -answer_groups[answer][0],
            ),
        )
        result.append(int(answer_groups[best_answer][0]))
    return result


def majority_choices(groups):
    return choose_from_scores(groups, np.zeros((len(groups), 4), dtype=np.float32), 1.0)


def group_correctness(answers, references):
    """Training/validation only; caller supplies aligned references explicitly."""
    gold = [parse_answer(reference["answer"]) for reference in references]
    return np.asarray([[answer is not None and answer == target for answer in row]
                       for row, target in zip(answers, gold)], dtype=bool)


def accuracy(choices, correct, groups=None, hard_only=False):
    mask = [len(group) >= 2 for group in groups] if hard_only else [True] * len(choices)
    indices = [i for i, include in enumerate(mask) if include]
    return 100 * sum(bool(correct[i, choices[i]]) for i in indices) / len(indices)


def answer_group_sizes(groups):
    return [Counter({key: len(indices) for key, indices in row.items()}) for row in groups]
