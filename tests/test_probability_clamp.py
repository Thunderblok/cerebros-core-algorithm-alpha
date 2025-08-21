import numpy as np
import pytest

from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component import (
    DenseLateralConnectivity,
)
from cerebros.utils.probability import clamp_probability


def test_clamp_probability_basic_cases():
    """Utility still clamps genuine probability-like values into [0,1]."""
    cases = [(-1.2, 0.0), (-0.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.2, 1.0)]
    for raw, expected in cases:
        assert clamp_probability(raw) == expected


def test_clamp_probability_nan_coerces_to_zero():
    assert clamp_probability(float("nan")) == 0.0


def test_dense_lateral_connectivity_intensity_unclamped():
    """p_lateral_connection now represents an intensity λ and is not clamped."""
    dlc_hi = DenseLateralConnectivity(p_lateral_connection=5.0)
    assert dlc_hi.p_lateral_connection == 5.0
    assert dlc_hi.raw_p_lateral_connection == 5.0

    dlc_neg = DenseLateralConnectivity(p_lateral_connection=-3.0)
    # Negative intensity is stored as-is (legacy callers can still introspect) but
    # sampling should yield zero multiplicity.
    assert dlc_neg.p_lateral_connection == -3.0
    assert dlc_neg.sample_lateral_connection_multiplicity(0) == 0


def test_update_intensity_unclamped():
    dlc = DenseLateralConnectivity(p_lateral_connection=0.2)
    dlc.update_p_lateral_connection(2.5)
    assert dlc.p_lateral_connection == 2.5
    dlc.update_p_lateral_connection(-0.5)
    assert dlc.p_lateral_connection == -0.5
    # Negative gives zero multiplicity
    assert dlc.sample_lateral_connection_multiplicity(0) == 0


def test_sample_lateral_connection_multiplicity_fractional_part(monkeypatch):
    """Verify floor + Bernoulli fractional logic for multiplicity sampling."""
    dlc = DenseLateralConnectivity(p_lateral_connection=2.25, p_lateral_connection_decay=lambda _: 1.0)

    # Force random draw < fractional part (0.25) so we get extra = 1
    monkeypatch.setattr(np.random, "random", lambda: 0.2)
    m1 = dlc.sample_lateral_connection_multiplicity(0)
    assert m1 == 3  # floor(2.25)=2 + 1

    # Force random draw > fractional part so extra = 0
    monkeypatch.setattr(np.random, "random", lambda: 0.3)
    m2 = dlc.sample_lateral_connection_multiplicity(0)
    assert m2 == 2


def test_negative_intensity_yields_zero_multiplicity():
    dlc = DenseLateralConnectivity(p_lateral_connection=-0.1)
    assert dlc.sample_lateral_connection_multiplicity(0) == 0


def test_select_connection_or_not_respects_max_consecutive_with_intensity():
    """High intensity should trigger connections until max_consecutive gating resets."""
    def decay_one(_):
        return 1.0

    dlc = DenseLateralConnectivity(
        p_lateral_connection=10.0,  # very high intensity => multiplicity >= 10
        p_lateral_connection_decay=decay_one,
        max_consecutive_lateral_connections=3,
    )

    draws = [dlc.select_connection_or_not(0) for _ in range(10)]
    # Expect pattern True True True False repeating (legacy gating semantics)
    for i in range(0, len(draws), 4):
        window = draws[i:i+4]
        if len(window) < 4:
            break
        assert window[:3] == [True, True, True]
        assert window[3] is False

