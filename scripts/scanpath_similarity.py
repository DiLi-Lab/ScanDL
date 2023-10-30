from __future__ import annotations

import textdistance


def levenshtein_distance(gt: list[int], pred: list[int]) -> float:
    return textdistance.levenshtein.distance(gt, pred)


def levenshtein_similarity(gt: list[int], pred: list[int]) -> float:
    return textdistance.levenshtein.similarity(gt, pred)


def levenshtein_normalized_similarity(gt: list[int], pred: list[int]) -> float:
    return textdistance.levenshtein.normalized_similarity(gt, pred)


def levenshtein_normalized_distance(gt: list[int], pred: list[int]) -> float:
    return textdistance.levenshtein.normalized_distance(gt, pred)
