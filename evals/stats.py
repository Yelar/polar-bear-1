"""
Stats helpers for evaluation.
"""

import math
import random
from typing import Iterable, List, Tuple, Dict


def _to_list(values: Iterable) -> List:
    return list(values) if values is not None else []


def compute_accuracy_strict(correct: Iterable[bool], valid: Iterable[bool]) -> float:
    """Strict accuracy: invalid counts as incorrect."""
    correct_list = _to_list(correct)
    valid_list = _to_list(valid)
    if not correct_list:
        return 0.0
    if len(valid_list) != len(correct_list):
        valid_list = [True] * len(correct_list)
    correct_count = sum(1 for c, v in zip(correct_list, valid_list) if v and c)
    return correct_count / len(correct_list)


def compute_accuracy_valid_only(correct: Iterable[bool], valid: Iterable[bool]) -> float:
    """Accuracy computed only over valid responses."""
    correct_list = _to_list(correct)
    valid_list = _to_list(valid)
    if not correct_list:
        return 0.0
    if len(valid_list) != len(correct_list):
        valid_list = [True] * len(correct_list)
    valid_count = sum(1 for v in valid_list if v)
    if valid_count == 0:
        return 0.0
    correct_count = sum(1 for c, v in zip(correct_list, valid_list) if v and c)
    return correct_count / valid_count


def compute_invalid_rate(valid: Iterable[bool]) -> float:
    """Invalid rate as a fraction of total responses."""
    valid_list = _to_list(valid)
    if not valid_list:
        return 0.0
    invalid = sum(1 for v in valid_list if not v)
    return invalid / len(valid_list)


def bootstrap_paired_diff(
    baseline: List[float],
    treatment: List[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Bootstrap paired difference; returns mean, ci_low, ci_high, p_better."""
    if not baseline or not treatment or len(baseline) != len(treatment):
        return 0.0, 0.0, 0.0, 0.5

    random.seed(seed)
    n = len(baseline)
    diffs = []
    for _ in range(n_boot):
        idxs = [random.randint(0, n - 1) for _ in range(n)]
        diff = sum(treatment[i] - baseline[i] for i in idxs) / n
        diffs.append(diff)

    diffs.sort()
    mean = sum(diffs) / len(diffs)
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]
    p_better = sum(1 for d in diffs if d > 0) / len(diffs)
    return mean, lo, hi, p_better


def mcnemar_test(correct_a: List[bool], correct_b: List[bool]) -> Tuple[float, float, float, float]:
    """McNemar's test (chi-square approx) for paired nominal data."""
    if len(correct_a) != len(correct_b) or not correct_a:
        return 0.0, 1.0, 0.0, 0.0
    b = sum(1 for a, b_ in zip(correct_a, correct_b) if a and not b_)
    c = sum(1 for a, b_ in zip(correct_a, correct_b) if not a and b_)
    if b + c == 0:
        return 0.0, 1.0, float(b), float(c)
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    pval = math.erfc(math.sqrt(chi2 / 2))
    return float(chi2), float(pval), float(b), float(c)
