"""Shared CIE xy distance and coordinate-wise goal tolerance helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

CIE_TOLERANCE = 0.005


def cie_within_coordinate_tolerance(
    cie: Sequence[float],
    target: Sequence[float],
    tolerance: float = CIE_TOLERANCE,
) -> bool:
    """Return true only when both CIE coordinates are within the independent tolerance."""
    if len(cie) != 2 or len(target) != 2:
        raise ValueError("CIE and target must each contain exactly two coordinates")
    if tolerance <= 0:
        raise ValueError("CIE tolerance must be positive")
    return all(abs(float(actual) - float(expected)) <= tolerance for actual, expected in zip(cie, target))


def cie_euclidean_distance(cie: Sequence[float], target: Sequence[float]) -> float:
    """Return Euclidean CIE distance for ranking and reporting, not goal acceptance."""
    if len(cie) != 2 or len(target) != 2:
        raise ValueError("CIE and target must each contain exactly two coordinates")
    return math.dist(tuple(float(value) for value in cie), tuple(float(value) for value in target))
