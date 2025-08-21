"""Utility helpers for probability parameters.

These helpers centralize defensive handling of user-provided probability
hyperparameters so that any value passed through public APIs is coerced into
the closed interval [0.0, 1.0]. This prevents subtle bugs (e.g., silently
using probabilities > 1 that break Bernoulli draws) and guards against NaNs.
"""
from __future__ import annotations

from math import isnan
from typing import Union

Number = Union[int, float]


def clamp_probability(p: Number, *, lower: float = 0.0, upper: float = 1.0,
                      nan_policy: str = "coerce_zero") -> float:
    """Clamp a numeric value into [lower, upper] (defaults to probability range).

    Args:
        p: The candidate probability-like value.
        lower: Minimum allowed value (inclusive); defaults to 0.0.
        upper: Maximum allowed value (inclusive); defaults to 1.0.
        nan_policy: How to handle NaN inputs.
            - "coerce_zero" (default): return 0.0
            - "raise": raise a ValueError
            - "ignore": return the original NaN (NOT recommended)

    Returns:
        A float within [lower, upper].
    """
    try:
        x = float(p)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"Cannot convert {p!r} to float for probability clamp") from exc

    if isnan(x):
        if nan_policy == "coerce_zero":
            return 0.0
        if nan_policy == "raise":  # pragma: no cover - rare path
            raise ValueError("NaN probability encountered with nan_policy='raise'")
        # nan_policy == "ignore": fall through
        return x

    if x < lower:
        return lower
    if x > upper:
        return upper
    return x


__all__ = ["clamp_probability"]
