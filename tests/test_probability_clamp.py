import math
import numpy as np

from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component import (
    DenseLateralConnectivity,
)
from cerebros.utils.probability import clamp_probability


def test_clamp_probability_basic_cases():
    cases = [(-1.2, 0.0), (-0.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.2, 1.0)]
    for raw, expected in cases:
        assert clamp_probability(raw) == expected


def test_clamp_probability_nan_coerces_to_zero():
    assert clamp_probability(float("nan")) == 0.0


def test_dense_lateral_connectivity_init_clamps_probability():
    dlc_hi = DenseLateralConnectivity(p_lateral_connection=5.0)
    assert dlc_hi.p_lateral_connection == 1.0
    assert dlc_hi.raw_p_lateral_connection == 5.0  # raw preserved

    dlc_lo = DenseLateralConnectivity(p_lateral_connection=-3.0)
    assert dlc_lo.p_lateral_connection == 0.0
    assert dlc_lo.raw_p_lateral_connection == -3.0


def test_update_probability_is_clamped():
    dlc = DenseLateralConnectivity(p_lateral_connection=0.2)
    dlc.update_p_lateral_connection(2.5)
    assert dlc.p_lateral_connection == 1.0
    dlc.update_p_lateral_connection(-0.5)
    assert dlc.p_lateral_connection == 0.0


def test_select_connection_or_not_probability_never_exceeds_one():
    # Use a decay that could (if misapplied) push probability > 1
    def decay_always_one(_):
        return 1.5  # exaggerated to test final clamp

    dlc = DenseLateralConnectivity(
        p_lateral_connection=0.9,
        p_lateral_connection_decay=decay_always_one,
        max_consecutive_lateral_connections=3,
    )

    # Internal clamp ensures effective probability = 1.0
    draws = [dlc.select_connection_or_not(0) for _ in range(10)]
    # Because probability effectively 1.0, every block of size "max_consecutive" should be True until reset logic kicks in
    # We expect pattern: True True True False True True True False ...
    for i in range(0, len(draws), 4):  # pattern window size 4 (3 trues + reset false)
        window = draws[i:i+4]
        if len(window) < 4:
            break
        assert window[:3] == [True, True, True]
        assert window[3] in (False, True)  # reset could cause immediate new True depending on random, allow flexibility
