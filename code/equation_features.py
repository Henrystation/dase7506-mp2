"""Ten input-only arithmetic consistency features for candidate selection.

This module never loads references, fits a model, runs candidate code, or calls
an external service. Arithmetic strings are checked with a bounded AST walker.
It uses only Python's standard library, NumPy and the original basic features.
"""
import ast
import math
import operator
import re

import numpy as np

from student_features import prepare_inputs


FEATURE_NAMES = (
    "log_checked_equations", "log_correct_equations", "log_incorrect_equations",
    "correct_equation_fraction", "incorrect_equation_fraction",
    "max_relative_equation_error", "mean_relative_equation_error",
    "last_checked_equation_correct", "last_checked_equation_error",
    "log_computed_results_matching_answer",
)
FRAC = re.compile(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.Pow: operator.pow}


def _number(text):
    if text is None:
        return None
    try:
        if "/" in text:
            a, b = text.split("/")
            return float(a.replace(",", "")) / float(b.replace(",", ""))
        return float(text.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def safe_arithmetic(expression):
    """Return a finite bounded arithmetic value or None for unsupported input."""
    if len(expression) > 150 or not re.search(r"\d", expression):
        return None
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 70:
            return None

        def visit(node):
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
                left, right = visit(node.left), visit(node.right)
                if (abs(left) > 1e15 or abs(right) > 1e15
                        or isinstance(node.op, ast.Pow) and abs(right) > 8):
                    raise ValueError("arithmetic bound exceeded")
                return OPERATORS[type(node.op)](left, right)
            raise ValueError("unsupported syntax")

        value = float(visit(tree))
        return value if math.isfinite(value) and abs(value) < 1e16 else None
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


def normalize_math(text):
    text = text.replace(",", "").replace("−", "-").replace("×", "*").replace("÷", "/")
    for _ in range(5):
        updated = FRAC.sub(r"((\1)/(\2))", text)
        if updated == text:
            break
        text = updated
    text = re.sub(r"\\(?:times|cdot)", "*", text)
    text = re.sub(r"\\(?:div)", "/", text)
    text = re.sub(r"\\(?:left|right)", "", text)
    text = re.sub(r"\\(?:text|mathrm)\{[^{}]*\}", " ", text)
    text = re.sub(r"(?<=\d)\s*%", "/100", text)
    text = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", text)
    return text.replace("^", "**")


def equation_stats(text, answer):
    normalized = normalize_math(text)
    checks, computed = [], []
    for match in re.finditer("=", normalized):
        left = normalized[max(0, match.start()-150):match.start()]
        right = normalized[match.end():match.end()+150]
        left_match = re.search(r"[\d.()+*/\- \t]+$", left)
        right_match = re.match(r"[\d.()+*/\- \t]+", right)
        if not left_match or not right_match:
            continue
        left_expression = left_match.group().strip()
        right_expression = right_match.group().strip().rstrip(".")
        if not any(op in left_expression for op in ("+", "-", "*", "/")):
            continue
        left_value = safe_arithmetic(left_expression)
        right_value = safe_arithmetic(right_expression)
        if left_value is None or right_value is None:
            continue
        error = abs(left_value-right_value)/max(1, abs(left_value), abs(right_value))
        computed.append(left_value)
        checks.append((error < 1e-4, min(1, error)))
    count = len(checks)
    good = sum(item[0] for item in checks)
    parsed = _number(answer)
    final_agree = (sum(abs(value-parsed) < 1e-4*max(1,abs(parsed)) for value in computed)
                   if parsed is not None else 0)
    return [math.log1p(count), math.log1p(good), math.log1p(count-good),
            good/max(count,1), (count-good)/max(count,1),
            max([item[1] for item in checks] or [0]),
            np.mean([item[1] for item in checks]) if count else 0,
            float(checks[-1][0]) if count else 0,
            checks[-1][1] if count else 0, math.log1p(final_agree)]


def prepare_equations(bank, arrays):
    """Basic 32 features plus ten equation features, independent of bank IDs."""
    answers, groups, embeddings, basic = prepare_inputs(bank, arrays)
    rows = [equation_stats(candidate["text"], answer)
            for row, row_answers in zip(bank, answers)
            for candidate, answer in zip(row["candidates"], row_answers)]
    equations = np.asarray(rows, dtype=np.float32).reshape(len(bank), 4, 10)
    return answers, groups, embeddings, np.concatenate([basic, equations], axis=-1)
