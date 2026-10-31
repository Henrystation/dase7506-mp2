"""Bounded, input-only arithmetic trajectory evidence for candidate selection.

The original 42 features are returned unchanged. Twenty additional features
describe possible numerical dependencies, error propagation, and symmetric
cross-candidate evidence. No reference answers, IDs, data files, training, code
execution, or external services are used here.

A matching operand is only *possible* provenance: equal numbers can represent
unrelated quantities. Multiple matching predecessors are therefore ambiguous,
and suppress recomputation and clean-chain evidence. Recomputed values are
features, never replacements for the official parsed answers. Semantic errors,
units, variables, and dependencies expressed in prose are outside this parser.

The extra analysis reads at most MAX_TEXT_CHARS characters, examines at most
MAX_EQUAL_SIGNS equal signs, and keeps at most MAX_STEPS steps per candidate.
Each expression is at most 150 characters and is checked by the existing
70-node bounded safe_arithmetic walker. The inherited 42-feature preparation
retains its existing full-text behavior.
"""

import ast
import copy
import math
import re

import numpy as np

from equation_features import normalize_math, prepare_equations, safe_arithmetic


FEATURE_NAMES = (
    "trajectory_log_steps",
    "trajectory_log_dependency_edges",
    "trajectory_dependency_step_fraction",
    "trajectory_ambiguous_operand_fraction",
    "trajectory_log_max_dependency_depth",
    "trajectory_tainted_step_fraction",
    "trajectory_terminal_step_fraction",
    "trajectory_answer_clean_terminal_support",
    "trajectory_answer_tainted_terminal_support",
    "trajectory_answer_incorrect_claim_support",
    "trajectory_answer_recomputed_terminal_support",
    "trajectory_answer_recomputed_terminal_conflict",
    "trajectory_terminal_recompute_available_fraction",
    "trajectory_terminal_claim_conflict_fraction",
    "trajectory_peer_clean_terminal_support_fraction",
    "trajectory_peer_clean_terminal_conflict_fraction",
    "trajectory_peer_recomputed_terminal_support_fraction",
    "trajectory_peer_recomputed_terminal_conflict_fraction",
    "trajectory_peer_answer_agreement_fraction",
    "trajectory_text_or_step_limit_hit",
)

MAX_TEXT_CHARS = 16_384
MAX_EQUAL_SIGNS = 48
MAX_STEPS = 32
MAX_EXPRESSION_CHARS = 150
_MATH_RUN = r"[\d.()+*/\- \t]+"
_LEFT_RUN = re.compile(_MATH_RUN + r"$")
_RIGHT_RUN = re.compile(_MATH_RUN)
_MATH_CHAR = re.compile(r"[\d.()+*/\- \t]")
_NUMERIC_RESULT = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:\s*/\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+))?\Z"
)


def _close(left, right):
    return abs(left - right) <= 1e-6 * max(1.0, abs(left), abs(right))


def _answer_number(answer):
    # Official answers are normalized integers or fractions, not arbitrary code.
    if answer is None or len(str(answer)) > MAX_EXPRESSION_CHARS:
        return None
    value = str(answer)
    return safe_arithmetic(value) if _NUMERIC_RESULT.fullmatch(value) else None


def _literal_value(node):
    """Treat a signed numeric literal as one operand, retaining its sign."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if (isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) in (int, float)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        return sign * float(node.operand.value)
    return None


def _operands(tree):
    """Return replaceable literals; never match a signless child twice."""
    pending, result = [tree], []
    while pending:
        node = pending.pop()
        value = _literal_value(node)
        if value is not None:
            result.append((node, value))
        else:
            pending.extend(ast.iter_child_nodes(node))
    return result


def _expressions(normalized, equal_position):
    """Extract complete adjacent numeric expressions, rejecting clipped runs."""
    left_start = max(0, equal_position - MAX_EXPRESSION_CHARS)
    right_end = equal_position + 1 + MAX_EXPRESSION_CHARS
    left_window = normalized[left_start:equal_position]
    right_window = normalized[equal_position + 1:right_end]
    left_match = _LEFT_RUN.search(left_window)
    right_match = _RIGHT_RUN.match(right_window)
    if left_match is None or right_match is None:
        return None

    raw_start = left_start + left_match.start()
    raw_end = equal_position + 1 + right_match.end()
    start = raw_start + len(left_match.group()) - len(left_match.group().lstrip())
    end = raw_end - len(right_match.group()) + len(right_match.group().rstrip())
    # Do not accept a suffix of a too-long expression or an identifier, nor a
    # numeric prefix immediately followed by an identifier/indexing operation.
    if start and (normalized[start - 1].isalnum()
                  or normalized[start - 1] in "_.[]"
                  or (raw_start > 0 and raw_start == left_start
                      and _MATH_CHAR.fullmatch(normalized[raw_start - 1]))):
        return None
    if end < len(normalized) and (
            normalized[end].isalnum() or normalized[end] in "_[]"
            or (raw_end == right_end and _MATH_CHAR.fullmatch(normalized[raw_end]))):
        return None

    left, right = left_match.group().strip(), right_match.group().strip()
    if not left or not _NUMERIC_RESULT.fullmatch(right):
        return None
    return left, right


class _ReplaceLiterals(ast.NodeTransformer):
    def __init__(self, replacements):
        self.replacements = replacements

    def visit(self, node):
        value = self.replacements.get(id(node))
        if value is not None:
            return ast.copy_location(ast.Constant(value=value), node)
        return super().visit(node)


def _analyze(text, answer):
    text = str(text)
    limited = len(text) > MAX_TEXT_CHARS
    normalized = normalize_math(text[:MAX_TEXT_CHARS])
    # A sentence-ending period is a boundary; decimal points remain intact.
    normalized = re.sub(r"(?<=\d)\.(?=\s|$)", "\n", normalized)
    steps, used_steps = [], set()
    operand_count = ambiguous_count = equal_count = 0

    for match in re.finditer("=", normalized):
        equal_count += 1
        if equal_count > MAX_EQUAL_SIGNS or len(steps) >= MAX_STEPS:
            limited = True
            break
        pair = _expressions(normalized, match.start())
        if pair is None:
            continue
        expression, claim_expression = pair
        computed = safe_arithmetic(expression)
        claimed = safe_arithmetic(claim_expression)
        if computed is None or claimed is None:
            continue
        tree = ast.parse(expression, mode="eval")
        if not any(isinstance(node, ast.BinOp) for node in ast.walk(tree)):
            continue

        # Copy once before identifying replaceable nodes, then re-use the
        # exact bounded safe evaluator after numerical substitution.
        tree = copy.deepcopy(tree)
        dependencies, replacements = set(), {}
        ambiguous = False
        for node, value in _operands(tree):
            operand_count += 1
            predecessors = [index for index, step in enumerate(steps)
                            if _close(value, step["claimed"])]
            if len(predecessors) > 1:
                ambiguous_count += 1
                ambiguous = True
            if predecessors:
                dependencies.update(predecessors)
                if len(predecessors) == 1:
                    previous = steps[predecessors[0]]["recomputed"]
                    if previous is not None:
                        replacements[id(node)] = previous

        used_steps.update(dependencies)
        correct = _close(computed, claimed)
        tainted = not correct or any(steps[i]["tainted"] for i in dependencies)
        uncertain = ambiguous or any(steps[i]["uncertain"] for i in dependencies)
        recomputed = None
        if not uncertain and all(steps[i]["recomputed"] is not None for i in dependencies):
            rewritten = _ReplaceLiterals(replacements).visit(tree)
            # unparse creates arithmetic text only; safe_arithmetic rechecks
            # length, AST size, operators, exponents, and finite result bounds.
            recomputed = safe_arithmetic(ast.unparse(rewritten))
        depth = 1 + max((steps[i]["depth"] for i in dependencies), default=0)
        steps.append({
            "claimed": claimed, "correct": correct, "tainted": tainted,
            "uncertain": uncertain, "recomputed": recomputed,
            "depth": depth, "dependencies": dependencies,
        })

    terminals = [step for i, step in enumerate(steps) if i not in used_steps]
    numeric_answer = _answer_number(answer)

    def matches(value):
        return numeric_answer is not None and value is not None and _close(value, numeric_answer)

    clean = [step for step in terminals if not step["tainted"] and not step["uncertain"]]
    available = [step for step in terminals if step["recomputed"] is not None]
    # Peer evidence must be tied to that peer's own final answer. Unrelated
    # terminal calculations cannot by themselves support another candidate.
    final_terminals = [step for step in terminals if matches(step["claimed"])]
    clean_values = [step["claimed"] for step in final_terminals
                    if not step["tainted"] and not step["uncertain"]]
    recomputed_values = [step["recomputed"] for step in final_terminals
                        if step["recomputed"] is not None]
    count, terminal_count = max(1, len(steps)), max(1, len(terminals))
    local = [
        math.log1p(len(steps)),
        math.log1p(sum(len(step["dependencies"]) for step in steps)),
        sum(bool(step["dependencies"]) for step in steps) / count,
        ambiguous_count / max(1, operand_count),
        math.log1p(max((step["depth"] for step in steps), default=0)),
        sum(step["tainted"] for step in steps) / count,
        len(terminals) / count,
        float(any(matches(step["claimed"]) for step in clean)),
        float(any(matches(step["claimed"]) and step["tainted"] for step in terminals)),
        float(any(matches(step["claimed"]) and not step["correct"] for step in steps)),
        float(any(matches(step["recomputed"]) for step in available)),
        float(any(matches(step["claimed"]) and not matches(step["recomputed"])
                  for step in available)),
        len(available) / terminal_count,
        sum(not _close(step["claimed"], step["recomputed"])
            for step in available) / terminal_count,
    ]
    return {
        "local": local, "answer": numeric_answer,
        "clean_values": clean_values, "recomputed_values": recomputed_values,
        "limited": float(limited),
    }


def _row_features(texts, answers):
    summaries = [_analyze(text, answer) for text, answer in zip(texts, answers)]
    result = []
    for index, summary in enumerate(summaries):
        target = summary["answer"]
        peers = [peer for j, peer in enumerate(summaries) if j != index]
        denominator = max(1, len(peers))
        cross = []
        for key in ("clean_values", "recomputed_values"):
            support = conflict = 0
            if target is not None:
                for peer in peers:
                    values = peer[key]
                    agrees = any(_close(target, value) for value in values)
                    support += agrees
                    conflict += bool(values) and not agrees
            cross.extend((support / denominator, conflict / denominator))
        agreement = sum(target is not None and peer["answer"] is not None
                        and _close(target, peer["answer"]) for peer in peers)
        result.append(summary["local"] + cross
                      + [agreement / denominator, summary["limited"]])
    return result


def prepare_trajectory(bank, arrays):
    """Return official answers/groups/embeddings and 42 + 20 input features.

    Candidate order only determines row alignment. All peer statistics are
    symmetric counts, so permuting candidates and embeddings permutes outputs.
    Neither row IDs nor candidate index fields are accessed by this module.
    """
    answers, groups, embeddings, existing = prepare_equations(bank, arrays)
    rows = [_row_features([candidate["text"] for candidate in row["candidates"]], row_answers)
            for row, row_answers in zip(bank, answers)]
    extra = np.asarray(rows, dtype=np.float32).reshape(len(bank), 4, len(FEATURE_NAMES))
    return answers, groups, embeddings, np.concatenate((existing, extra), axis=-1)


def _sanity():
    """Small fabricated examples only; no data loading or model fitting."""
    from itertools import permutations

    def local(text, answer):
        result = _analyze(text, answer)
        return dict(zip(FEATURE_NAMES[:14], result["local"]))

    good = "2+3=5; 5*4=20; #### 20"
    bad = "2+3=6; 6*4=24; #### 24"
    unrelated = "2+3=6; 10*2=20; #### 20"
    no_answer = "There is no marked numeric answer."
    g, b, u = local(good, "20"), local(bad, "24"), local(unrelated, "20")
    assert g["trajectory_answer_clean_terminal_support"] == 1
    assert g["trajectory_tainted_step_fraction"] == 0
    prose = local("We calculate 2 + 3 = 5 dollars. Then 5 * 4 = 20 dollars. #### 20", "20")
    assert prose["trajectory_answer_clean_terminal_support"] == 1
    for text in ("We have 2+3=5", "2+3=5 apples", "  2+3=5; #### 5"):
        assert local(text, "5")["trajectory_answer_clean_terminal_support"] == 1
    for text in ("identifier2+3=5", "2+3=5identifier", "call(2+3)=5"):
        assert local(text, "5")["trajectory_log_steps"] == 0
    assert b["trajectory_answer_tainted_terminal_support"] == 1
    assert b["trajectory_answer_recomputed_terminal_conflict"] == 1
    assert _analyze(bad, "24")["recomputed_values"] == [20.0]
    assert u["trajectory_answer_clean_terminal_support"] == 1
    assert u["trajectory_answer_tainted_terminal_support"] == 0

    ambiguous = local("2+3=5; 10/2=5; 5*4=20; #### 20", "20")
    assert ambiguous["trajectory_ambiguous_operand_fraction"] > 0
    assert ambiguous["trajectory_terminal_recompute_available_fraction"] == 0
    assert ambiguous["trajectory_answer_clean_terminal_support"] == 0
    signed = local("1-3=-2; -2*4=-8; #### -8", "-8")
    assert signed["trajectory_answer_clean_terminal_support"] == 1
    fraction = local("1/2=0.5; 0.5*3=1.5; #### 3/2", "3/2")
    assert fraction["trajectory_answer_clean_terminal_support"] == 1
    malicious = '__import__("os").system("never_run")=0; 2**999999=0; [1,2][0]=1; #### 0'
    attack = local(malicious, "0")
    assert attack["trajectory_log_steps"] == 0
    assert all(math.isfinite(value) for value in attack.values())
    assert _analyze("=" * (MAX_EQUAL_SIGNS + 1), None)["limited"] == 1
    assert _analyze("x" * (MAX_TEXT_CHARS + 1), None)["limited"] == 1
    assert local("1+" * 100 + "1=101; #### 101", "101")["trajectory_log_steps"] == 0

    texts = [good, bad, unrelated, no_answer]
    bank = [{"id": "fabricated", "question": "A fabricated arithmetic check.",
             "candidates": [{"index": i, "text": text} for i, text in enumerate(texts)]}]
    arrays = {"embeddings": np.arange(4 * 1024, dtype=np.float32).reshape(1, 4, 1024)}
    base = prepare_equations(bank, arrays)
    original = prepare_trajectory(bank, arrays)
    assert original[0] == [["20", "24", "20", None]]
    assert original[3].shape == (1, 4, 62)
    np.testing.assert_array_equal(original[3][:, :, :42], base[3])
    assert np.isfinite(original[3]).all()
    names = {name: index + 42 for index, name in enumerate(FEATURE_NAMES)}
    assert original[3][0, 0, names["trajectory_peer_recomputed_terminal_support_fraction"]] > 0
    assert original[3][0, 1, names["trajectory_peer_clean_terminal_conflict_fraction"]] > 0
    for order in permutations(range(4)):
        permuted_bank = [{"id": "different ignored ID", "question": bank[0]["question"],
                          "candidates": [{"index": 999 - i, "text": texts[j]}
                                         for i, j in enumerate(order)]}]
        permuted = prepare_trajectory(permuted_bank, {"embeddings": arrays["embeddings"][:, order, :]})
        np.testing.assert_array_equal(permuted[2], original[2][:, order, :])
        np.testing.assert_array_equal(permuted[3], original[3][:, order, :])
        assert permuted[0][0] == [original[0][0][j] for j in order]
    print("trajectory_features_v6 sanity passed: 20 extra / 62 total features; "
          "correct, tainted, unrelated, ambiguous, signed, fractional, malicious, "
          "bounded, unchanged-base, and all 24 candidate permutations checked.")


if __name__ == "__main__":
    _sanity()
